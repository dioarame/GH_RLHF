using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using NetMQ;
using NetMQ.Sockets;
using Rhino;
using System.Globalization;
using Newtonsoft.Json;
using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace GrasshopperZmqComponent
{
    /// <summary>
    /// ZMQ를 통해 상태와 보상 값을 Python으로 전송하는 컴포넌트
    /// </summary>
    public class ZmqStateSender : GH_Component, IDisposable
    {
        // GUID 상수 정의
        private static readonly Guid COMPONENT_GUID = new Guid("D8F6B21E-8A5C-4E78-9C9D-6F3A8B2E7D5A");

        // ZMQ 관련 변수
        private PushSocket zmqPushSocket = null;
        private string currentAddress = "";
        private bool isRunning = false;
        private bool disposed = false;
        private readonly object socketLock = new object();

        // 데이터 추적 변수
        private readonly Stopwatch uptimeStopwatch = new Stopwatch();
        private string lastState = null;
        private string lastReward = null;
        private string lastAction = null;
        private long messageCounter = 0;
        private long sendFailureCounter = 0;
        private int socketReconnectCount = 0;
        private bool? previousRunState = null;
        private readonly object startStopLock = new object();

        // 타이머 관련 변수
        private System.Timers.Timer healthCheckTimer = null;
        private readonly int healthCheckInterval = 5000; // 5초마다 소켓 상태 확인

        public ZmqStateSender()
            : base("ZMQ State Sender", "StateSender",
                "Grasshopper 상태와 보상 값을 ZMQ를 통해 Python으로 전송합니다.",
                "Extra", "Communication")
        {
            RhinoApp.WriteLine($"ZmqStateSender 인스턴스 생성됨: {InstanceGuid}");
            uptimeStopwatch.Start();
        }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBooleanParameter("Run", "R", "전송을 시작하려면 True로 설정", GH_ParamAccess.item, false);
            pManager.AddTextParameter("Address", "A", "Python ZMQ PULL 소켓 주소 (예: tcp://localhost:5557)", GH_ParamAccess.item, "tcp://localhost:5557");
            pManager.AddTextParameter("Current State", "State", "현재 상태 값 (쉼표로 구분된 실수 목록)", GH_ParamAccess.item);
            pManager.AddTextParameter("Current Reward", "Reward", "현재 보상 값 (실수)", GH_ParamAccess.item);
            pManager.AddTextParameter("Last Action", "Action", "마지막으로 적용된 액션 값 (선택 사항)", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Send Rate Limit", "Rate", "초당 최대 전송 횟수 (0=제한 없음, 기본값: 5)", GH_ParamAccess.item, 5);

            // 선택적 파라미터 설정
            pManager[2].Optional = true; // State는 선택적
            pManager[3].Optional = true; // Reward는 선택적
            pManager[4].Optional = true; // Action은 선택적
            pManager[5].Optional = true; // Rate는 선택적
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Status", "Status", "현재 연결 상태 및 메시지 통계", GH_ParamAccess.item);
            pManager.AddTextParameter("Last Sent", "Sent", "마지막으로 전송된 데이터", GH_ParamAccess.item);
        }

        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                string iconPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "Icons", "ZmqStateSender.png");
                if (File.Exists(iconPath))
                {
                    return new System.Drawing.Bitmap(iconPath);
                }
                return null; // 이미지를 찾지 못하면 기본 아이콘 사용
            }
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            if (disposed)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "컴포넌트가 폐기되었습니다.");
                return;
            }

            // 입력 파라미터 가져오기
            bool run = false;
            string address = "tcp://localhost:5557";
            string stateValue = null;
            string rewardValue = null;
            string actionValue = null;
            int rateLimit = 5;

            DA.GetData(0, ref run);
            DA.GetData(1, ref address);
            DA.GetData(2, ref stateValue);
            DA.GetData(3, ref rewardValue);
            DA.GetData(4, ref actionValue);
            DA.GetData(5, ref rateLimit);

            // 연결 상태 관리
            bool runStateChanged = (previousRunState == null || previousRunState != run);
            bool addressChanged = (currentAddress != address);

            if (runStateChanged || (run && addressChanged))
            {
                RhinoApp.WriteLine($"송신기 상태 변경: Run={run}, Address='{address}' (이전: {previousRunState}, '{currentAddress}')");

                if (run)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, $"{address}에 연결 시작 중...");
                    StartSender(address);
                }
                else
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "송신기 중지 중...");
                    StopSender();
                }

                previousRunState = run;
            }

            // 데이터 전송 처리
            string lastSentData = "No data sent";
            string statusMessage = GetStatusMessage();

            if (run && isRunning)
            {
                // 데이터 변경 여부 확인 (모든 값이 null이면 변경되지 않은 것으로 간주)
                bool hasStateChanged = (stateValue != null && stateValue != lastState);
                bool hasRewardChanged = (rewardValue != null && rewardValue != lastReward);
                bool hasActionChanged = (actionValue != null && actionValue != lastAction);
                bool shouldSendUpdate = hasStateChanged || hasRewardChanged || hasActionChanged;

                // 타임스탬프 및 데이터 생성
                if (shouldSendUpdate)
                {
                    Dictionary<string, object> dataPackage = new Dictionary<string, object>();
                    dataPackage["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    dataPackage["uptime_ms"] = uptimeStopwatch.ElapsedMilliseconds;

                    // 상태 값 처리
                    if (stateValue != null)
                    {
                        try
                        {
                            List<double> stateValues = ParseCommaSeparatedValues(stateValue);
                            dataPackage["state"] = stateValues;
                            lastState = stateValue;
                        }
                        catch (Exception ex)
                        {
                            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"상태 값 파싱 오류: {ex.Message}");
                            dataPackage["state_error"] = $"파싱 실패: {stateValue}";
                        }
                    }

                    // 보상 값 처리
                    if (rewardValue != null)
                    {
                        try
                        {
                            if (double.TryParse(rewardValue, NumberStyles.Any, CultureInfo.InvariantCulture, out double reward))
                            {
                                dataPackage["reward"] = reward;
                                lastReward = rewardValue;
                            }
                            else
                            {
                                dataPackage["reward_error"] = $"파싱 실패: {rewardValue}";
                            }
                        }
                        catch (Exception ex)
                        {
                            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"보상 값 파싱 오류: {ex.Message}");
                            dataPackage["reward_error"] = $"파싱 실패: {rewardValue}";
                        }
                    }

                    // 액션 값 처리
                    if (actionValue != null)
                    {
                        try
                        {
                            // 액션이 JSON 배열 형식인지 확인
                            if (actionValue.Trim().StartsWith("[") && actionValue.Trim().EndsWith("]"))
                            {
                                List<double> actionValues = JsonConvert.DeserializeObject<List<double>>(actionValue);
                                dataPackage["action"] = actionValues;
                            }
                            else if (actionValue.Contains(","))
                            {
                                // 쉼표로 구분된 목록인 경우
                                List<double> actionValues = ParseCommaSeparatedValues(actionValue);
                                dataPackage["action"] = actionValues;
                            }
                            else
                            {
                                // 단일 값인 경우
                                if (double.TryParse(actionValue, NumberStyles.Any, CultureInfo.InvariantCulture, out double action))
                                {
                                    dataPackage["action"] = new List<double> { action };
                                }
                                else
                                {
                                    dataPackage["action_error"] = $"파싱 실패: {actionValue}";
                                }
                            }
                            lastAction = actionValue;
                        }
                        catch (Exception ex)
                        {
                            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"액션 값 파싱 오류: {ex.Message}");
                            dataPackage["action_error"] = $"파싱 실패: {actionValue}";
                        }
                    }

                    // 메시지 ID 추가
                    dataPackage["msg_id"] = Interlocked.Increment(ref messageCounter);

                    // JSON 직렬화 및 전송
                    string jsonData = JsonConvert.SerializeObject(dataPackage);
                    bool sendSuccess = SendData(jsonData);

                    if (sendSuccess)
                    {
                        lastSentData = jsonData.Length > 100 ? jsonData.Substring(0, 97) + "..." : jsonData;
                    }
                    else
                    {
                        lastSentData = "전송 실패: " + (jsonData.Length > 80 ? jsonData.Substring(0, 77) + "..." : jsonData);
                    }
                }
                else
                {
                    lastSentData = "변경된 데이터 없음";
                }
            }
            else
            {
                lastSentData = "송신기 비활성화됨";
            }

            // 출력 설정
            statusMessage = GetStatusMessage();
            DA.SetData(0, statusMessage);
            DA.SetData(1, lastSentData);
        }

        /// <summary>
        /// 쉼표로 구분된 문자열을 실수 목록으로 파싱
        /// </summary>
        private List<double> ParseCommaSeparatedValues(string input)
        {
            List<double> result = new List<double>();

            if (string.IsNullOrWhiteSpace(input))
                return result;

            // 따옴표 제거
            if (input.Length > 1 && (input[0] == '"' || input[0] == '\'') && input[input.Length - 1] == input[0])
            {
                input = input.Substring(1, input.Length - 2);
            }

            string[] parts = input.Split(',');
            foreach (string part in parts)
            {
                string trimmed = part.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    if (double.TryParse(trimmed, NumberStyles.Any, CultureInfo.InvariantCulture, out double value))
                    {
                        result.Add(value);
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// ZMQ PUSH 소켓을 통해 데이터 전송
        /// </summary>
        private bool SendData(string jsonData)
        {
            lock (socketLock)
            {
                if (zmqPushSocket == null || zmqPushSocket.IsDisposed)
                {
                    Interlocked.Increment(ref sendFailureCounter);
                    RhinoApp.WriteLine("송신기 소켓이 없거나 폐기됨. 재연결 필요.");
                    return false;
                }

                try
                {
                    zmqPushSocket.SendFrame(jsonData, false);
                    return true;
                }
                catch (TerminatingException)
                {
                    // NetMQ 종료 중 예외
                    Interlocked.Increment(ref sendFailureCounter);
                    return false;
                }
                catch (ObjectDisposedException)
                {
                    // 소켓이 이미 폐기됨
                    Interlocked.Increment(ref sendFailureCounter);
                    return false;
                }
                catch (Exception ex)
                {
                    // 기타 전송 오류
                    Interlocked.Increment(ref sendFailureCounter);
                    RhinoApp.WriteLine($"데이터 전송 중 오류: {ex.Message}");
                    return false;
                }
            }
        }

        /// <summary>
        /// ZMQ 송신기 시작
        /// </summary>
        private void StartSender(string address)
        {
            lock (startStopLock)
            {
                if (isRunning)
                {
                    if (currentAddress != address)
                    {
                        RhinoApp.WriteLine($"송신기가 실행 중이지만 주소가 '{address}'(으)로 변경됨. 재시작 중...");
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "주소가 변경되었습니다. 송신기를 재시작합니다...");
                        StopSenderInternal();
                    }
                    else
                    {
                        RhinoApp.WriteLine($"송신기가 이미 {address}에서 실행 중입니다.");
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "송신기가 이미 실행 중입니다.");
                        return;
                    }
                }

                try
                {
                    // PUSH 소켓 초기화
                    zmqPushSocket = new PushSocket();

                    // 소켓 옵션 설정
                    zmqPushSocket.Options.SendHighWatermark = 1000;
                    zmqPushSocket.Options.Linger = TimeSpan.FromMilliseconds(500);
                    zmqPushSocket.Options.ReconnectInterval = TimeSpan.FromSeconds(1);
                    zmqPushSocket.Options.TcpKeepalive = true;
                    zmqPushSocket.Options.TcpKeepaliveIdle = TimeSpan.FromSeconds(30);
                    zmqPushSocket.Options.TcpKeepaliveInterval = TimeSpan.FromSeconds(5);

                    RhinoApp.WriteLine($"PUSH 소켓 연결 시도: {address}");
                    zmqPushSocket.Connect(address);

                    currentAddress = address;
                    isRunning = true;

                    // 상태 확인 타이머 시작
                    StartHealthCheckTimer();

                    // 테스트 메시지 전송
                    SendTestMessage();

                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, $"{address}에 연결됨");
                    RhinoApp.WriteLine($"ZMQ PUSH 송신기가 {address}에 연결됨");
                }
                catch (Exception ex)
                {
                    RhinoApp.WriteLine($"송신기 시작 중 오류: {ex.Message}");
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"연결 오류: {ex.Message}");
                    StopSenderInternal();
                }
            }
        }

        /// <summary>
        /// 상태 확인 타이머 시작
        /// </summary>
        private void StartHealthCheckTimer()
        {
            if (healthCheckTimer != null)
            {
                healthCheckTimer.Stop();
                healthCheckTimer.Dispose();
            }

            healthCheckTimer = new System.Timers.Timer(healthCheckInterval);
            healthCheckTimer.Elapsed += (sender, e) => CheckSocketHealth();
            healthCheckTimer.Start();
        }

        /// <summary>
        /// 소켓 상태 확인
        /// </summary>
        private void CheckSocketHealth()
        {
            if (!isRunning) return;

            lock (socketLock)
            {
                if (zmqPushSocket == null || zmqPushSocket.IsDisposed)
                {
                    RhinoApp.WriteLine("상태 확인: 소켓이 없거나 폐기됨. 재연결 시도 중...");

                    RhinoApp.InvokeOnUiThread((Action)(() =>
                    {
                        if (!disposed)
                        {
                            StopSenderInternal();
                            Thread.Sleep(1000);
                            StartSender(currentAddress);
                        }
                    }));
                    return;
                }

                // 주기적인 Keepalive 메시지 전송
                try
                {
                    Dictionary<string, object> healthCheckData = new Dictionary<string, object>
                    {
                        ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                        ["uptime_ms"] = uptimeStopwatch.ElapsedMilliseconds,
                        ["type"] = "health_check",
                        ["msg_id"] = Interlocked.Increment(ref messageCounter)
                    };

                    string jsonData = JsonConvert.SerializeObject(healthCheckData);
                    zmqPushSocket.SendFrame(jsonData, false);

                    // 로그 간소화: 매 10번째 상태 확인마다 출력
                    if (messageCounter % 10 == 0)
                    {
                        RhinoApp.WriteLine($"상태 확인: Keepalive 메시지 전송됨 (#{messageCounter})");
                    }
                }
                catch (Exception ex)
                {
                    RhinoApp.WriteLine($"상태 확인 중 오류: {ex.Message}");
                    socketReconnectCount++;

                    RhinoApp.InvokeOnUiThread((Action)(() =>
                    {
                        if (!disposed)
                        {
                            StopSenderInternal();
                            Thread.Sleep(1000);
                            StartSender(currentAddress);
                        }
                    }));
                }
            }
        }

        /// <summary>
        /// 테스트 메시지 전송
        /// </summary>
        private void SendTestMessage()
        {
            Dictionary<string, object> testData = new Dictionary<string, object>
            {
                ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                ["uptime_ms"] = uptimeStopwatch.ElapsedMilliseconds,
                ["type"] = "connection_test",
                ["msg_id"] = Interlocked.Increment(ref messageCounter)
            };

            string jsonData = JsonConvert.SerializeObject(testData);
            SendData(jsonData);
        }

        /// <summary>
        /// 송신기 중지 (외부 호출용)
        /// </summary>
        private void StopSender()
        {
            lock (startStopLock)
            {
                StopSenderInternal();
            }
        }

        /// <summary>
        /// 송신기 내부 중지 함수
        /// </summary>
        private void StopSenderInternal()
        {
            if (!isRunning && zmqPushSocket == null && healthCheckTimer == null)
            {
                return;
            }

            RhinoApp.WriteLine("송신기 리소스 정리 중...");
            isRunning = false;

            // 타이머 중지
            if (healthCheckTimer != null)
            {
                healthCheckTimer.Stop();
                healthCheckTimer.Dispose();
                healthCheckTimer = null;
            }

            // 소켓 정리
            lock (socketLock)
            {
                if (zmqPushSocket != null)
                {
                    try
                    {
                        if (!zmqPushSocket.IsDisposed)
                        {
                            zmqPushSocket.Options.Linger = TimeSpan.Zero;
                            zmqPushSocket.Disconnect(currentAddress);
                            zmqPushSocket.Close();
                            zmqPushSocket.Dispose();
                        }
                    }
                    catch (Exception ex)
                    {
                        RhinoApp.WriteLine($"소켓 정리 중 오류: {ex.Message}");
                    }
                    finally
                    {
                        zmqPushSocket = null;
                    }
                }
            }

            currentAddress = "";
            RhinoApp.WriteLine("송신기 리소스 정리 완료");
        }

        /// <summary>
        /// 상태 메시지 생성
        /// </summary>
        private string GetStatusMessage()
        {
            if (!isRunning)
            {
                return "연결 끊김";
            }

            return $"연결됨 (PUSH, {currentAddress}) | 전송: {messageCounter} | 실패: {sendFailureCounter} | 재연결: {socketReconnectCount}";
        }

        public override Guid ComponentGuid
        {
            get { return COMPONENT_GUID; }
        }

        // --- IDisposable 구현 ---
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed) return;

            RhinoApp.WriteLine($"ZmqStateSender 인스턴스 폐기 중 ({disposing}): {InstanceGuid}");

            if (disposing)
            {
                // 관리 리소스 정리
                StopSender();
            }

            disposed = true;
        }

        ~ZmqStateSender()
        {
            Dispose(false);
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            RhinoApp.WriteLine($"인스턴스 제거됨: {InstanceGuid.ToString().Substring(0, 6)}...");
            Dispose();
            base.RemovedFromDocument(document);
        }
    }
}