// 필요한 using 문
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Special;
using Grasshopper.Kernel.Types;
using NetMQ;
using NetMQ.Sockets;
using Rhino;
using System.Globalization;
using Newtonsoft.Json;
using System.IO;
using System.Reflection;

namespace GrasshopperZmqComponent
{
    public class ZmqListener : GH_Component, IDisposable
    {
        // --- 멤버 변수 ---
        private PullSocket instancePullSocket = null;
        private CancellationTokenSource instanceCts = null;
        private Task instanceListenTask = null;
        private bool instanceIsRunning = false;
        private bool taskActuallyRunning = false;
        private string instanceCurrentAddress = "";
        private readonly object instanceLatestValueLock = new object();
        private string instanceLatestValue = "Listener stopped.";
        private List<Guid> instanceTargetGuids = new List<Guid>();
        private bool disposed = false;
        private bool? previousRunState = null;
        private readonly object startStopLock = new object();
        private int receiveTimeoutMs = 500; // 수신 타임아웃 증가
        private int consecutiveTimeoutLimit = 200; // 연속 타임아웃 한계 증가
        private int reconnectAttemptDelay = 1000; // 재연결 시도 간격 (ms)
        private int socketCleanupDelay = 300; // 소켓 정리 지연시간 (ms)
        private int healthCheckInterval = 10; // 상태 확인 간격 (초)
        private DateTime lastHealthCheck = DateTime.MinValue;
        private int healthCheckTimeoutCount = 0; // 연속 타임아웃 카운터
        private int socketReconnectCount = 0; // 소켓 재연결 카운터

        // --- 생성자 ---
        public ZmqListener()
          : base("ZMQ Action Applier", "ZMQ Listener",
                 "Connects to a Python ZMQ PUSH server and receives action arrays to apply to target Number Sliders.",
                 "Extra", "Communication")
        {
            RhinoApp.WriteLine($"ZmqListener Instance Created: {InstanceGuid}");
        }

        // --- ComponentGuid ---
        public override Guid ComponentGuid => new Guid("8A29DD67-42B0-4A84-9A39-C22C08842855");

        // --- 입력 파라미터 등록 ---
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBooleanParameter("Run", "R", "Connect and start listening when true", GH_ParamAccess.item, false);
            pManager.AddTextParameter("Address", "A", "Python ZMQ server address (e.g., tcp://localhost:5556)", GH_ParamAccess.item, "tcp://localhost:5556");
            pManager.AddTextParameter("Target Slider GUIDs", "GUIDs", "Instance GUIDs of Number Sliders to control", GH_ParamAccess.list);
            pManager[2].Optional = true;
        }

        // --- 출력 파라미터 등록 ---
        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Last Received Action", "Action", "Last raw action message received via ZMQ", GH_ParamAccess.item);
            pManager.AddTextParameter("Connection Status", "Status", "Current connection status and health information", GH_ParamAccess.item);
        }
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                string iconPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "Icons", "ZmqListener.png");
                if (File.Exists(iconPath))
                {
                    return new System.Drawing.Bitmap(iconPath);
                }
                return null; // 이미지를 찾지 못하면 기본 아이콘 사용
            }
        }

        // --- 컴포넌트 메인 실행 로직 ---
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            if (disposed) { AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Component has been disposed."); return; }

            bool run = false;
            string address = "tcp://localhost:5556";
            List<string> guidStrings = new List<string>();

            if (!DA.GetData(0, ref run)) return;
            if (!DA.GetData(1, ref address)) return;
            DA.GetDataList(2, guidStrings);

            // GUID 목록 처리
            instanceTargetGuids.Clear();
            foreach (string guidStr in guidStrings)
            {
                if (!string.IsNullOrWhiteSpace(guidStr) && Guid.TryParse(guidStr, out Guid parsedGuid))
                { instanceTargetGuids.Add(parsedGuid); }
                else if (!string.IsNullOrWhiteSpace(guidStr))
                { AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Failed to parse GUID: {guidStr}"); }
            }

            // 상태 변경 감지
            bool runStateChanged = (previousRunState == null || previousRunState != run);
            bool addressChanged = (instanceCurrentAddress != address);

            if (runStateChanged || (run && addressChanged))
            {
                RhinoApp.WriteLine($"State Changed: Run={run}, Address='{address}' (Previous: {previousRunState}, '{instanceCurrentAddress}')");

                if (run)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, $"Starting connection to {address}...");
                    StartListening_Instance(address);
                }
                else
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Stopping listener...");
                    StopListening_Instance();
                }
            }

            // 주기적인 상태 확인 및 상태 업데이트
            if (run && instanceIsRunning)
            {
                // 주기적 상태 확인 (healthCheckInterval 초마다)
                TimeSpan timeSinceLastCheck = DateTime.Now - lastHealthCheck;
                if (timeSinceLastCheck.TotalSeconds >= healthCheckInterval)
                {
                    lastHealthCheck = DateTime.Now;
                    CheckConnectionHealth();
                }
            }

            previousRunState = run;

            // 출력 값 설정
            string outputValue;
            lock (instanceLatestValueLock) { outputValue = instanceLatestValue; }
            DA.SetData(0, outputValue);

            // 연결 상태 출력 (상세 정보 포함)
            string statusMsg = "Disconnected";
            if (instanceIsRunning)
            {
                statusMsg = taskActuallyRunning ?
                    $"Connected - PULL mode | Reconnects: {socketReconnectCount} | Timeouts: {healthCheckTimeoutCount}" :
                    "Connecting...";
            }
            DA.SetData(1, statusMsg);
        }

        // --- 연결 상태 확인 ---
        private void CheckConnectionHealth()
        {
            // 소켓과 태스크 상태 확인
            if (instancePullSocket == null || instancePullSocket.IsDisposed)
            {
                RhinoApp.WriteLine("Health Check: Socket is null or disposed. Reconnecting...");
                StopListening_Internal();
                Thread.Sleep(reconnectAttemptDelay);
                StartListening_Instance(instanceCurrentAddress);
                return;
            }

            if (instanceListenTask == null || instanceListenTask.IsFaulted || instanceListenTask.IsCanceled)
            {
                RhinoApp.WriteLine($"Health Check: Listener task issue - " +
                    $"null: {instanceListenTask == null}, " +
                    $"faulted: {instanceListenTask?.IsFaulted ?? false}, " +
                    $"canceled: {instanceListenTask?.IsCanceled ?? false}");
                StopListening_Internal();
                Thread.Sleep(reconnectAttemptDelay);
                StartListening_Instance(instanceCurrentAddress);
                return;
            }

            RhinoApp.WriteLine($"Health Check: Socket OK, Task OK, Timeouts: {healthCheckTimeoutCount}");
        }

        // --- 인스턴스 리스너 시작 함수 ---
        private void StartListening_Instance(string address)
        {
            lock (startStopLock)
            {
                if (instanceIsRunning || taskActuallyRunning)
                {
                    if (instanceCurrentAddress != address)
                    {
                        RhinoApp.WriteLine($"Listener running, but address changed to '{address}'. Restarting...");
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Address changed. Restarting listener...");
                        StopListening_Internal();
                    }
                    else
                    {
                        RhinoApp.WriteLine($"Listener start requested for {address}, but already running.");
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Listener already running.");
                        return;
                    }
                }

                instanceIsRunning = true;
                instanceCurrentAddress = address;
                healthCheckTimeoutCount = 0;

                RhinoApp.WriteLine($"Instance {InstanceGuid.ToString().Substring(0, 4)}: Connecting to {address} using PULL socket...");
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, $"Connecting to {address}...");
                RequestExpireSolution();

                try
                {
                    // PULL 소켓 초기화 (최적화 버전)
                    instancePullSocket = new PullSocket();

                    // 소켓 옵션 최적화
                    instancePullSocket.Options.ReceiveHighWatermark = 1000;  // 낮춘 값
                    instancePullSocket.Options.Linger = TimeSpan.FromMilliseconds(200);  // 낮춘 값
                    instancePullSocket.Options.ReconnectInterval = TimeSpan.FromMilliseconds(1000);
                    instancePullSocket.Options.Backlog = 100;  // 백로그 제한

                    // TCP Keepalive 설정
                    instancePullSocket.Options.TcpKeepalive = true;
                    instancePullSocket.Options.TcpKeepaliveIdle = TimeSpan.FromSeconds(30);
                    instancePullSocket.Options.TcpKeepaliveInterval = TimeSpan.FromSeconds(5);

                    RhinoApp.WriteLine($"연결 시도 (PULL): {address}");
                    instancePullSocket.Connect(address);

                    RhinoApp.WriteLine($"Socket Options: HWM={instancePullSocket.Options.ReceiveHighWatermark}, " +
                                     $"Linger={instancePullSocket.Options.Linger.TotalMilliseconds}ms, " +
                                     $"Backlog={instancePullSocket.Options.Backlog}");

                    instanceCts = new CancellationTokenSource();
                    CancellationToken token = instanceCts.Token;

                    // 리스닝 태스크 시작
                    instanceListenTask = Task.Run(() => ListenLoop(token, address), token);

                    RhinoApp.WriteLine($"Instance {InstanceGuid.ToString().Substring(0, 4)}: Connection initiated to {address}. " +
                                     $"Listening task starting (PULL mode).");
                }
                catch (Exception ex)
                {
                    RhinoApp.WriteLine($"[ERROR] Instance {InstanceGuid.ToString().Substring(0, 4)}: Failed to initiate connection: {ex.Message}");
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Connect Error: {ex.Message}. Is Python server running?");
                    StopListening_Internal();
                    instanceIsRunning = false;
                    taskActuallyRunning = false;
                    instanceCurrentAddress = "";
                    RequestExpireSolution();
                }
            }
        }

        // --- 리스너 루프 ---
        private void ListenLoop(CancellationToken token, string expectedAddress)
        {
            taskActuallyRunning = true;
            string instanceIdPrefix = $"PULL Listener {Thread.CurrentThread.ManagedThreadId}";
            RhinoApp.WriteLine($"{instanceIdPrefix}: Started. Waiting for messages from {expectedAddress}...");
            int messageCounter = 0;
            bool reportedConnected = false;
            int consecutiveTimeouts = 0;

            try
            {
                while (!token.IsCancellationRequested)
                {
                    if (instanceCurrentAddress != expectedAddress)
                    { RhinoApp.WriteLine($"{instanceIdPrefix}: Address changed. Stopping loop."); break; }

                    string message = null;
                    bool received = false;
                    List<double> actionValues = null;

                    try
                    {
                        if (instancePullSocket == null || instancePullSocket.IsDisposed)
                        {
                            RhinoApp.WriteLine($"{instanceIdPrefix}: Socket is null or disposed. Exiting loop.");
                            break;
                        }

                        // 소켓 상태 확인 로직 (주기적으로)
                        if (consecutiveTimeouts % 20 == 0)
                        {
                            RhinoApp.WriteLine($"{instanceIdPrefix}: Socket state check - " +
                                            $"Connected={!instancePullSocket.IsDisposed}, " +
                                            $"Timeouts={consecutiveTimeouts}");
                        }

                        // 메시지 수신 시도
                        received = instancePullSocket.TryReceiveFrameString(TimeSpan.FromMilliseconds(receiveTimeoutMs), out message);

                        if (received)
                        {
                            consecutiveTimeouts = 0;
                            healthCheckTimeoutCount = 0; // 전체 상태 카운터도 리셋

                            if (!reportedConnected)
                            {
                                RhinoApp.WriteLine($"{instanceIdPrefix}: Connected and received first message.");
                                RhinoApp.InvokeOnUiThread((Action)(() => {
                                    if (!disposed) AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, $"Connected to {expectedAddress}.");
                                }));
                                reportedConnected = true;
                                RequestExpireSolution();
                            }

                            // 디버깅 로그 추가 - 100개 메시지마다 로그 출력
                            if (messageCounter % 100 == 0)
                            {
                                RhinoApp.WriteLine($"[DEBUG] 메시지 #{messageCounter}: '{message.Substring(0, Math.Min(20, message.Length))}...'");
                            }
                        }
                        else
                        {
                            consecutiveTimeouts++;
                            healthCheckTimeoutCount++; // 전체 상태 카운터도 증가

                            // 연속 타임아웃 로그 출력 빈도 조정
                            if (consecutiveTimeouts % 20 == 0)
                            {
                                if (reportedConnected)
                                {
                                    RhinoApp.WriteLine($"{instanceIdPrefix}: No message received for ~{(receiveTimeoutMs * consecutiveTimeouts) / 1000.0:F1}s...");
                                }
                                else
                                {
                                    RhinoApp.WriteLine($"{instanceIdPrefix}: Still waiting for connection/first message...");
                                    RhinoApp.InvokeOnUiThread((Action)(() => {
                                        if (!disposed) AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Waiting for connection to {expectedAddress}...");
                                    }));
                                    RequestExpireSolution();
                                }
                            }

                            // 연속 타임아웃이 한계를 초과하면 소켓 재연결
                            if (consecutiveTimeouts > consecutiveTimeoutLimit)
                            {
                                RhinoApp.WriteLine($"{instanceIdPrefix}: Too many consecutive timeouts ({consecutiveTimeouts}). Reconnecting socket...");
                                socketReconnectCount++;

                                // UI 스레드에서 재연결 요청
                                RhinoApp.InvokeOnUiThread((Action)(() =>
                                {
                                    if (!disposed)
                                    {
                                        StopListening_Internal();
                                        Thread.Sleep(reconnectAttemptDelay); // 재연결 전 대기 시간 증가
                                        StartListening_Instance(expectedAddress);
                                    }
                                }));

                                break; // 현재 루프 종료 (새 루프가 시작될 것임)
                            }

                            continue;
                        }

                        if (message != null)
                        {
                            messageCounter++;
                            try
                            {
                                // JSON 파싱 시도
                                if (message.Trim().StartsWith("[") && message.Trim().EndsWith("]"))
                                {
                                    actionValues = JsonConvert.DeserializeObject<List<double>>(message);
                                }
                                else
                                {
                                    if (double.TryParse(message, NumberStyles.Any, CultureInfo.InvariantCulture, out double singleValue))
                                    { actionValues = new List<double> { singleValue }; }
                                    else { throw new JsonException($"Cannot parse as number or JSON array"); }
                                }

                                if (actionValues == null) throw new JsonException("Deserialized list is null.");

                                // 값 로깅 추가 (100개마다)
                                if (messageCounter % 100 == 0)
                                {
                                    RhinoApp.WriteLine($"Message #{messageCounter}: Values={string.Join(", ", actionValues)}");
                                }

                                List<double> valuesCopy = new List<double>(actionValues);
                                string receivedMessageCopy = message;
                                RhinoApp.InvokeOnUiThread((Action)delegate {
                                    UpdateSlidersFromUIThread(valuesCopy, receivedMessageCopy, messageCounter);
                                });
                            }
                            catch (JsonException jsonEx)
                            {
                                RhinoApp.WriteLine($"{instanceIdPrefix} [{messageCounter}]: JSON Error: {jsonEx.Message}");
                                lock (instanceLatestValueLock) { instanceLatestValue = $"JSON Error: {jsonEx.Message}"; }
                                RequestExpireSolution();
                                continue;
                            }
                        }
                    }
                    catch (OperationCanceledException) { break; }
                    catch (TerminatingException) { break; }
                    catch (ObjectDisposedException) { RhinoApp.WriteLine($"{instanceIdPrefix}: Socket disposed, stopping loop."); break; }
                    catch (Exception ex)
                    {
                        if (!token.IsCancellationRequested)
                        {
                            RhinoApp.WriteLine($"{instanceIdPrefix}: Listener loop error: {ex.GetType().Name} - {ex.Message}");
                            string errorMsg = $"Listener Error: {ex.Message}";
                            RhinoApp.InvokeOnUiThread((Action)(() => {
                                if (!disposed) AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, errorMsg);
                            }));
                        }
                        break;
                    }
                }
            }
            finally
            {
                RhinoApp.WriteLine($"{instanceIdPrefix}: Loop finished.");
                taskActuallyRunning = false;
                RhinoApp.InvokeOnUiThread((Action)(() => {
                    if (!disposed)
                    {
                        instanceIsRunning = false;
                        if (reportedConnected) AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Listener stopped.");
                        else AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Listener stopped (error or never connected).");
                        RequestExpireSolution();
                    }
                }));
            }
        }

        // --- 슬라이더 업데이트 (UI 스레드) ---
        private void UpdateSlidersFromUIThread(List<double> values, string rawMessage, int counter)
        {
            if (disposed) return;

            lock (instanceLatestValueLock)
            {
                instanceLatestValue = rawMessage.Length > 100 ?
                    rawMessage.Substring(0, 97) + "..." :
                    rawMessage;
            }

            int guidCount = instanceTargetGuids.Count;
            if (guidCount == 0) return;

            GH_Document doc = OnPingDocument();
            if (doc == null) return;

            int valuesCount = values.Count;
            int slidersToUpdate = Math.Min(guidCount, valuesCount);
            bool changed = false;

            for (int i = 0; i < slidersToUpdate; i++)
            {
                Guid currentGuid = instanceTargetGuids[i];
                double currentValue = values[i];
                IGH_DocumentObject sliderObj = doc.FindObject<IGH_DocumentObject>(currentGuid, true);

                if (sliderObj is GH_NumberSlider slider)
                {
                    try
                    {
                        decimal newValueDecimal = Convert.ToDecimal(currentValue);

                        // 슬라이더 타입 확인 및 반올림
                        int decimalPlaces = slider.Slider.DecimalPlaces;
                        if (decimalPlaces == 0) // 정수 슬라이더인 경우
                        {
                            newValueDecimal = decimal.Round(newValueDecimal);
                        }
                        else if (decimalPlaces > 0)
                        {
                            newValueDecimal = decimal.Round(newValueDecimal, decimalPlaces);
                        }

                        // 슬라이더 범위 체크
                        if (newValueDecimal < slider.Slider.Minimum) newValueDecimal = slider.Slider.Minimum;
                        if (newValueDecimal > slider.Slider.Maximum) newValueDecimal = slider.Slider.Maximum;

                        // 값이 변경된 경우에만 업데이트
                        if (slider.Slider.Value != newValueDecimal)
                        {
                            // 로그 주기 제한 (1000개 메시지마다 또는 초기 메시지)
                            if (counter % 1000 == 0 || counter < 10)
                            {
                                RhinoApp.WriteLine($"Slider {i} ({currentGuid.ToString().Substring(0, 8)}): {slider.Slider.Value} -> {newValueDecimal}");
                            }
                            slider.SetSliderValue(newValueDecimal);
                            changed = true;
                        }
                    }
                    catch (Exception ex)
                    {
                        RhinoApp.WriteLine($"Slider update error: {ex.Message}");
                    }
                }
                else
                {
                    // 첫 10개 메시지에서만 출력하여 로그 스팸 감소
                    if (counter < 10)
                    {
                        RhinoApp.WriteLine($"GUID {currentGuid.ToString().Substring(0, 8)} is not a slider or not found");
                    }
                }
            }

            if (changed)
            {
                // 로그 주기 제한
                if (counter % 100 == 0)
                {
                    RhinoApp.WriteLine($"Updated {slidersToUpdate} sliders. (Message #{counter})");
                }
                ExpireSolution(true);
            }
        }

        // --- 리스너 중지 함수 (외부 호출용) ---
        private void StopListening_Instance()
        {
            lock (startStopLock) { StopListening_Internal(); }
        }

        // --- 내부 리소스 정리 함수 ---
        private void StopListening_Internal()
        {
            string instanceIdPrefix = $"Instance {InstanceGuid.ToString().Substring(0, 4)}";

            if (!instanceIsRunning && !taskActuallyRunning && instancePullSocket == null && instanceListenTask == null)
            {
                return;
            }

            RhinoApp.WriteLine($"{instanceIdPrefix}: Stopping listener resources...");
            instanceIsRunning = false;

            // 1. CancellationTokenSource 취소
            if (instanceCts != null)
            {
                try
                {
                    if (!instanceCts.IsCancellationRequested)
                        instanceCts.Cancel();
                }
                catch { }
            }

            // 2. 리스너 태스크 종료 대기
            Task taskToWait = instanceListenTask;
            if (taskToWait != null)
            {
                RhinoApp.WriteLine($"{instanceIdPrefix}: Waiting for listener task ({taskToWait.Id}) to complete...");
                try
                {
                    // 최대 500ms 대기
                    taskToWait.Wait(TimeSpan.FromMilliseconds(500));
                }
                catch { }

                RhinoApp.WriteLine($"{instanceIdPrefix}: Task {(taskToWait.IsCompleted ? "completed" : "still running")}");
            }
            instanceListenTask = null;
            taskActuallyRunning = false;

            // 3. 소켓 정리 전 짧은 지연
            Thread.Sleep(socketCleanupDelay);

            // 4. PULL 소켓 정리
            PullSocket pullSocketToClose = instancePullSocket;
            instancePullSocket = null;
            if (pullSocketToClose != null && !pullSocketToClose.IsDisposed)
            {
                RhinoApp.WriteLine($"{instanceIdPrefix}: Closing and disposing PULL socket...");
                try
                {
                    // 소켓 정리 순서 최적화
                    pullSocketToClose.Options.Linger = TimeSpan.Zero;

                    // 연결 종료 후 소켓 닫기/정리
                    try { pullSocketToClose.Disconnect(instanceCurrentAddress); }
                    catch (Exception ex) { RhinoApp.WriteLine($"Error disconnecting: {ex.Message}"); }

                    try { pullSocketToClose.Close(); }
                    catch (Exception ex) { RhinoApp.WriteLine($"Error closing socket: {ex.Message}"); }

                    try { pullSocketToClose.Dispose(); }
                    catch (Exception ex) { RhinoApp.WriteLine($"Error disposing socket: {ex.Message}"); }

                    RhinoApp.WriteLine($"{instanceIdPrefix}: PULL socket closed and disposed.");
                }
                catch (Exception ex) { RhinoApp.WriteLine($"{instanceIdPrefix}: Error closing/disposing PULL socket: {ex.Message}"); }
            }

            // 5. CancellationTokenSource 정리
            CancellationTokenSource ctsToDispose = instanceCts;
            instanceCts = null;
            if (ctsToDispose != null) { try { ctsToDispose.Dispose(); } catch { } }

            instanceCurrentAddress = "";
            RhinoApp.WriteLine($"{instanceIdPrefix}: Listener resources stopped/cleaned.");

            RhinoApp.InvokeOnUiThread((Action)(() => {
                if (!disposed) AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Listener stopped.");
                RequestExpireSolution();
            }));

            // 가비지 컬렉션 요청 (선택적)
            try { GC.Collect(); } catch { }
        }

        // --- 포트 가용성 확인 함수 ---
        private bool IsPortAvailable(int port)
        {
            try
            {
                using (var socket = new System.Net.Sockets.TcpClient())
                {
                    try
                    {
                        socket.Connect("127.0.0.1", port);
                        return false; // 연결 성공 = 포트 사용 중
                    }
                    catch (System.Net.Sockets.SocketException)
                    {
                        return true; // 연결 실패 = 포트 사용 가능
                    }
                }
            }
            catch
            {
                return false; // 다른 오류 = 알 수 없음, 안전하게 false 반환
            }
        }

        // --- IDisposable 구현 ---
        public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
        protected virtual void Dispose(bool disposing)
        {
            if (disposed) return;
            RhinoApp.WriteLine($"Disposing ZmqListener Instance ({disposing}): {InstanceGuid}");
            if (disposing) { /* 관리 리소스 */ }
            lock (startStopLock) { StopListening_Internal(); }
            disposed = true;
        }
        ~ZmqListener() { Dispose(false); }

        // --- 컴포넌트 제거/문서 닫기 ---
        public override void RemovedFromDocument(GH_Document document)
        {
            RhinoApp.WriteLine($"Instance Removed: {InstanceGuid.ToString().Substring(0, 6)}...");
            Dispose();
            base.RemovedFromDocument(document);
        }

        // --- UI 업데이트 요청 ---
        private void RequestExpireSolution()
        {
            GH_Document doc = OnPingDocument();
            if (!disposed && doc != null)
            {
                try
                {
                    RhinoApp.InvokeOnUiThread((Action)(() =>
                    {
                        if (!disposed && doc.FindObject<IGH_DocumentObject>(this.InstanceGuid, true) != null)
                        { ExpireSolution(true); }
                    }));
                }
                catch (Exception ex) { RhinoApp.WriteLine($"Error requesting expire solution: {ex.Message}"); }
            }
        }
    } // 클래스 종료
} // 네임스페이스 종료