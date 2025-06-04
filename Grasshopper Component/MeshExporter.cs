using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino;
using Rhino.Geometry;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System.Reflection;

namespace GrasshopperZmqComponent
{
    /// <summary>
    /// 그래스호퍼 메쉬 데이터를 JSON 또는 glTF 형식으로 내보내는 컴포넌트
    /// RLHF 시스템에서 인간 피드백을 위한 3D 메쉬 데이터 추출에 사용됨
    /// </summary>
    public class MeshExporter : GH_Component, IDisposable
    {
        // GUID 상수
        private static readonly Guid COMPONENT_GUID = new Guid("B47D2E5F-8C7A-4B89-B3D8-CE54A793F42A");

        // ZMQ 관련 변수
        private readonly ZmqHandler zmqHandler;
        private string zmqCurrentAddress = "";
        private bool zmqIsRunning = false;
        private bool disposed = false;

        // 메시 데이터 캐시
        private readonly Dictionary<string, string> meshCache = new Dictionary<string, string>();
        private readonly object cacheLock = new object();

        public MeshExporter()
            : base("Mesh Exporter", "MeshExport",
                "메쉬 데이터를 JSON 또는 glTF 형식으로 내보냅니다",
                "Extra", "Communication")
        {
            RhinoApp.WriteLine($"MeshExporter 인스턴스 생성됨: {InstanceGuid}");
            zmqHandler = new ZmqHandler(ProcessZmqRequest);
        }

        public override Guid ComponentGuid => COMPONENT_GUID;

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBooleanParameter("Run", "R", "데이터 내보내기 및 ZMQ 리스너 활성화", GH_ParamAccess.item, false);
            pManager.AddTextParameter("Address", "A", "ZMQ 소켓 주소 (예: tcp://localhost:5558)", GH_ParamAccess.item, "tcp://localhost:5558");
            pManager.AddMeshParameter("Mesh", "M", "내보낼 메쉬 데이터", GH_ParamAccess.tree);
            pManager.AddTextParameter("Export Format", "F", "내보내기 형식 (json 또는 gltf)", GH_ParamAccess.item, "json");
            pManager.AddTextParameter("Export Path", "P", "내보내기 파일 경로 (선택 사항)", GH_ParamAccess.item, "");
            pManager.AddIntegerParameter("Export ID", "ID", "내보내기 식별자 (선택 사항)", GH_ParamAccess.item, 0);

            // 선택적 파라미터 설정
            pManager[4].Optional = true;
            pManager[5].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Status", "S", "현재 상태 정보", GH_ParamAccess.item);
            pManager.AddTextParameter("Data", "D", "직렬화된 메쉬 데이터", GH_ParamAccess.item);
            pManager.AddTextParameter("Export Path", "P", "내보내기 경로", GH_ParamAccess.item);
        }
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                string iconPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "Icons", "MeshExporter.png");
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
            string address = "tcp://localhost:5558";
            GH_Structure<GH_Mesh> meshTree = new GH_Structure<GH_Mesh>();
            string exportFormat = "json";
            string exportPath = "";
            int exportId = 0;

            if (!DA.GetData(0, ref run)) return;
            if (!DA.GetData(1, ref address)) return;
            if (!DA.GetDataTree(2, out meshTree)) return;
            if (!DA.GetData(3, ref exportFormat)) return;
            DA.GetData(4, ref exportPath);
            DA.GetData(5, ref exportId);

            // 형식 유효성 검사
            exportFormat = exportFormat.ToLower().Trim();
            if (exportFormat != "json" && exportFormat != "gltf")
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "지원되지 않는 형식입니다. 'json' 또는 'gltf'만 지원됩니다.");
                exportFormat = "json"; // 기본값으로 설정
            }

            // ZMQ 리스너 상태 관리
            bool runStateChanged = (run != zmqIsRunning);
            bool addressChanged = (address != zmqCurrentAddress);

            if (runStateChanged || (run && addressChanged))
            {
                if (run)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, $"{address}에 연결 시작 중...");
                    zmqHandler.Start(address);
                    zmqIsRunning = true;
                    zmqCurrentAddress = address;
                }
                else
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "ZMQ 리스너 중지 중...");
                    zmqHandler.Stop();
                    zmqIsRunning = false;
                    zmqCurrentAddress = "";
                }
            }

            // 메쉬 데이터 처리
            string meshData = "{}";
            string status = run ? "실행 중" : "중지됨";

            if (run && !meshTree.IsEmpty)
            {
                try
                {
                    // 데이터셋 키 생성 (exportId 또는 현재 시간 사용)
                    string datasetKey = exportId > 0 ? exportId.ToString() : DateTime.Now.Ticks.ToString();

                    // 메쉬 데이터 처리
                    string exportData = ProcessMeshTree(meshTree, exportFormat, datasetKey);

                    // 출력 데이터 설정 (너무 큰 경우 요약 정보만 제공)
                    if (exportData.Length > 1000)
                    {
                        meshData = $"{{\"format\": \"{exportFormat}\", \"datasetKey\": \"{datasetKey}\", \"size\": {exportData.Length} bytes}}";
                    }
                    else
                    {
                        meshData = exportData;
                    }

                    // 메쉬 캐시에 저장
                    lock (cacheLock)
                    {
                        meshCache[datasetKey] = exportData;
                    }

                    // 파일로 내보내기
                    if (!string.IsNullOrEmpty(exportPath))
                    {
                        string fileName = Path.GetFileName(exportPath);
                        string directory = Path.GetDirectoryName(exportPath);

                        if (string.IsNullOrEmpty(fileName))
                        {
                            fileName = $"mesh_export_{datasetKey}.{exportFormat}";
                        }

                        if (string.IsNullOrEmpty(directory))
                        {
                            directory = Path.GetTempPath();
                        }

                        // 디렉토리 존재 확인
                        if (!Directory.Exists(directory))
                        {
                            Directory.CreateDirectory(directory);
                        }

                        string fullPath = Path.Combine(directory, fileName);
                        File.WriteAllText(fullPath, exportData, Encoding.UTF8);

                        exportPath = fullPath;
                        status = $"파일 내보내기 완료: {fullPath}";
                    }
                    else
                    {
                        status = $"메쉬 데이터 처리 완료 (ZMQ 요청 대기 중)";
                    }
                }
                catch (Exception ex)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"메쉬 처리 중 오류 발생: {ex.Message}");
                    status = $"오류: {ex.Message}";
                }
            }

            // 출력 설정
            DA.SetData(0, status);
            DA.SetData(1, meshData);
            DA.SetData(2, exportPath);
        }

        /// <summary>
        /// 메쉬 트리를 처리하여 지정된 형식의 문자열로 반환
        /// </summary>
        private string ProcessMeshTree(GH_Structure<GH_Mesh> meshTree, string format, string datasetKey)
        {
            switch (format.ToLower())
            {
                case "json":
                    return ConvertToJson(meshTree, datasetKey);
                case "gltf":
                    return ConvertToGltf(meshTree, datasetKey);
                default:
                    return "{}";
            }
        }

        /// <summary>
        /// 메쉬 트리를 JSON 형식으로 변환
        /// </summary>
        private string ConvertToJson(GH_Structure<GH_Mesh> meshTree, string datasetKey)
        {
            var jsonObj = new
            {
                datasetKey = datasetKey,
                format = "json",
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                meshes = new List<object>()
            };

            var meshList = (List<object>)jsonObj.meshes;

            // 모든 경로의 메쉬 처리
            foreach (GH_Path path in meshTree.Paths)
            {
                var meshesAtPath = meshTree[path];

                foreach (GH_Mesh ghMesh in meshesAtPath)
                {
                    if (ghMesh == null) continue;

                    // GH_Mesh에서 Mesh 추출
                    Mesh mesh = ghMesh.Value;
                    if (mesh == null || !mesh.IsValid) continue;

                    var vertices = new List<double[]>();
                    var faces = new List<int[]>();
                    var normals = new List<double[]>();

                    // 정점 추출
                    foreach (Point3f v in mesh.Vertices)
                    {
                        vertices.Add(new double[] { v.X, v.Y, v.Z });
                    }

                    // 면 추출
                    foreach (MeshFace face in mesh.Faces)
                    {
                        if (face.IsTriangle)
                        {
                            faces.Add(new int[] { face.A, face.B, face.C });
                        }
                        else if (face.IsQuad)
                        {
                            faces.Add(new int[] { face.A, face.B, face.C, face.D });
                        }
                    }

                    // 법선 추출 (있는 경우)
                    if (mesh.Normals.Count > 0)
                    {
                        foreach (Vector3f n in mesh.Normals)
                        {
                            normals.Add(new double[] { n.X, n.Y, n.Z });
                        }
                    }

                    // 메쉬 데이터 추가
                    var meshData = new
                    {
                        path = path.ToString(),
                        vertexCount = vertices.Count,
                        faceCount = faces.Count,
                        vertices = vertices,
                        faces = faces,
                        normals = normals.Count > 0 ? normals : null,
                        boundingBox = new
                        {
                            min = new double[] { mesh.GetBoundingBox(true).Min.X, mesh.GetBoundingBox(true).Min.Y, mesh.GetBoundingBox(true).Min.Z },
                            max = new double[] { mesh.GetBoundingBox(true).Max.X, mesh.GetBoundingBox(true).Max.Y, mesh.GetBoundingBox(true).Max.Z }
                        }
                    };

                    meshList.Add(meshData);
                }
            }

            return JsonConvert.SerializeObject(jsonObj, Formatting.None);
        }

        /// <summary>
        /// 메쉬 트리를 glTF 형식으로 변환
        /// </summary>
        private string ConvertToGltf(GH_Structure<GH_Mesh> meshTree, string datasetKey)
        {
            // glTF 구조 초기화
            var gltf = new
            {
                asset = new { version = "2.0", generator = "GrasshopperZmqComponent.MeshExporter" },
                scene = 0,
                scenes = new[] { new { nodes = new List<int>() } },
                nodes = new List<object>(),
                meshes = new List<object>(),
                accessors = new List<object>(),
                bufferViews = new List<object>(),
                buffers = new List<object>()
            };

            var sceneNodes = (List<int>)gltf.scenes[0].nodes;
            var nodesList = (List<object>)gltf.nodes;
            var meshesList = (List<object>)gltf.meshes;
            var accessorsList = (List<object>)gltf.accessors;
            var bufferViewsList = (List<object>)gltf.bufferViews;
            var buffersList = (List<object>)gltf.buffers;

            // 바이너리 데이터를 저장할 메모리 스트림
            using (MemoryStream binStream = new MemoryStream())
            {
                int nodeIndex = 0;
                int meshIndex = 0;
                int accessorIndex = 0;
                int bufferViewIndex = 0;

                // 모든 경로의 메쉬 처리
                foreach (GH_Path path in meshTree.Paths)
                {
                    var meshesAtPath = meshTree[path];

                    foreach (GH_Mesh ghMesh in meshesAtPath)
                    {
                        if (ghMesh == null) continue;

                        // GH_Mesh에서 Mesh 추출
                        Mesh mesh = ghMesh.Value;
                        if (mesh == null || !mesh.IsValid) continue;

                        // 노드 생성
                        nodesList.Add(new
                        {
                            name = $"mesh_{nodeIndex}",
                            mesh = meshIndex
                        });

                        sceneNodes.Add(nodeIndex);
                        nodeIndex++;

                        // 정점 데이터
                        var vertices = new float[mesh.Vertices.Count * 3];
                        for (int i = 0; i < mesh.Vertices.Count; i++)
                        {
                            vertices[i * 3] = (float)mesh.Vertices[i].X;
                            vertices[i * 3 + 1] = (float)mesh.Vertices[i].Y;
                            vertices[i * 3 + 2] = (float)mesh.Vertices[i].Z;
                        }

                        // 법선 데이터
                        var normals = new float[mesh.Normals.Count * 3];
                        for (int i = 0; i < mesh.Normals.Count; i++)
                        {
                            normals[i * 3] = (float)mesh.Normals[i].X;
                            normals[i * 3 + 1] = (float)mesh.Normals[i].Y;
                            normals[i * 3 + 2] = (float)mesh.Normals[i].Z;
                        }

                        // 인덱스 데이터 (삼각형으로 변환)
                        var indices = new List<int>();
                        foreach (MeshFace face in mesh.Faces)
                        {
                            if (face.IsTriangle)
                            {
                                indices.Add(face.A);
                                indices.Add(face.B);
                                indices.Add(face.C);
                            }
                            else if (face.IsQuad)
                            {
                                // 쿼드를 두 개의 삼각형으로 분할
                                indices.Add(face.A);
                                indices.Add(face.B);
                                indices.Add(face.C);

                                indices.Add(face.A);
                                indices.Add(face.C);
                                indices.Add(face.D);
                            }
                        }

                        // 바이너리 데이터 추가
                        long vertexBufferOffset = binStream.Position;
                        byte[] vertexBytes = new byte[vertices.Length * 4];
                        Buffer.BlockCopy(vertices, 0, vertexBytes, 0, vertexBytes.Length);
                        binStream.Write(vertexBytes, 0, vertexBytes.Length);

                        long normalBufferOffset = binStream.Position;
                        byte[] normalBytes = new byte[normals.Length * 4];
                        Buffer.BlockCopy(normals, 0, normalBytes, 0, normalBytes.Length);
                        binStream.Write(normalBytes, 0, normalBytes.Length);

                        long indexBufferOffset = binStream.Position;
                        byte[] indexBytes = new byte[indices.Count * 4];
                        Buffer.BlockCopy(indices.ToArray(), 0, indexBytes, 0, indexBytes.Length);
                        binStream.Write(indexBytes, 0, indexBytes.Length);

                        // 버퍼 뷰 추가
                        bufferViewsList.Add(new
                        {
                            buffer = 0,
                            byteOffset = vertexBufferOffset,
                            byteLength = vertexBytes.Length,
                            target = 34962  // ARRAY_BUFFER
                        });
                        int vertexBufferViewIndex = bufferViewIndex++;

                        bufferViewsList.Add(new
                        {
                            buffer = 0,
                            byteOffset = normalBufferOffset,
                            byteLength = normalBytes.Length,
                            target = 34962  // ARRAY_BUFFER
                        });
                        int normalBufferViewIndex = bufferViewIndex++;

                        bufferViewsList.Add(new
                        {
                            buffer = 0,
                            byteOffset = indexBufferOffset,
                            byteLength = indexBytes.Length,
                            target = 34963  // ELEMENT_ARRAY_BUFFER
                        });
                        int indexBufferViewIndex = bufferViewIndex++;

                        // 엑세서 추가
                        accessorsList.Add(new
                        {
                            bufferView = vertexBufferViewIndex,
                            byteOffset = 0,
                            componentType = 5126,  // FLOAT
                            count = mesh.Vertices.Count,
                            type = "VEC3",
                            min = new float[] {
                                (float)mesh.GetBoundingBox(true).Min.X,
                                (float)mesh.GetBoundingBox(true).Min.Y,
                                (float)mesh.GetBoundingBox(true).Min.Z
                            },
                            max = new float[] {
                                (float)mesh.GetBoundingBox(true).Max.X,
                                (float)mesh.GetBoundingBox(true).Max.Y,
                                (float)mesh.GetBoundingBox(true).Max.Z
                            }
                        });
                        int positionAccessorIndex = accessorIndex++;

                        accessorsList.Add(new
                        {
                            bufferView = normalBufferViewIndex,
                            byteOffset = 0,
                            componentType = 5126,  // FLOAT
                            count = mesh.Normals.Count,
                            type = "VEC3"
                        });
                        int normalAccessorIndex = accessorIndex++;

                        accessorsList.Add(new
                        {
                            bufferView = indexBufferViewIndex,
                            byteOffset = 0,
                            componentType = 5125,  // UNSIGNED_INT
                            count = indices.Count,
                            type = "SCALAR"
                        });
                        int indicesAccessorIndex = accessorIndex++;

                        // 메쉬 추가
                        meshesList.Add(new
                        {
                            primitives = new[] {
                                new {
                                    attributes = new {
                                        POSITION = positionAccessorIndex,
                                        NORMAL = normalAccessorIndex
                                    },
                                    indices = indicesAccessorIndex,
                                    mode = 4  // TRIANGLES
                                }
                            }
                        });

                        meshIndex++;
                    }
                }

                // 바이너리 데이터를 Base64 문자열로 변환
                byte[] binData = binStream.ToArray();
                string base64Data = Convert.ToBase64String(binData);

                // 버퍼 추가
                buffersList.Add(new
                {
                    byteLength = binData.Length,
                    uri = $"data:application/octet-stream;base64,{base64Data}"
                });
            }

            // glTF 구조를 JSON 문자열로 직렬화
            return JsonConvert.SerializeObject(gltf, Formatting.None);
        }

        /// <summary>
        /// ZMQ 요청 처리
        /// </summary>
        private string ProcessZmqRequest(string request)
        {
            try
            {
                // 요청 파싱
                var requestObj = JsonConvert.DeserializeObject<Dictionary<string, object>>(request);
                string requestType = requestObj.ContainsKey("request") ? requestObj["request"].ToString() : "";
                string datasetKey = requestObj.ContainsKey("datasetKey") ? requestObj["datasetKey"].ToString() : "";

                switch (requestType.ToLower())
                {
                    case "get_mesh":
                        if (!string.IsNullOrEmpty(datasetKey))
                        {
                            lock (cacheLock)
                            {
                                if (meshCache.ContainsKey(datasetKey))
                                {
                                    // 캐시에서 메쉬 데이터 반환
                                    return meshCache[datasetKey];
                                }
                            }
                        }
                        return JsonConvert.SerializeObject(new { status = "error", message = "메쉬 데이터를 찾을 수 없음" });

                    case "list_datasets":
                        lock (cacheLock)
                        {
                            var datasets = new List<string>(meshCache.Keys);
                            return JsonConvert.SerializeObject(new { status = "success", datasets = datasets });
                        }

                    case "ping":
                        return JsonConvert.SerializeObject(new { status = "success", message = "MeshExporter가 응답합니다." });

                    default:
                        return JsonConvert.SerializeObject(new { status = "error", message = "알 수 없는 요청 유형" });
                }
            }
            catch (Exception ex)
            {
                return JsonConvert.SerializeObject(new { status = "error", message = ex.Message });
            }
        }

        /// <summary>
        /// IDisposable 인터페이스 구현
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed) return;

            if (disposing)
            {
                zmqHandler.Dispose();
            }

            disposed = true;
        }

        /// <summary>
        /// 소멸자
        /// </summary>
        ~MeshExporter()
        {
            Dispose(false);
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            Dispose();
            base.RemovedFromDocument(document);
        }
    }

    /// <summary>
    /// ZMQ 통신을 담당하는 헬퍼 클래스
    /// </summary>
    public class ZmqHandler : IDisposable
    {
        // ZMQ 관련 변수
        private ResponseSocket socket = null;
        private CancellationTokenSource cts = null;
        private Task listenTask = null;
        private string currentAddress = "";
        private bool isRunning = false;
        private readonly object lockObject = new object();
        private bool disposed = false;

        // 요청 처리 함수
        private readonly Func<string, string> requestHandler;

        /// <summary>
        /// 생성자
        /// </summary>
        public ZmqHandler(Func<string, string> requestHandler)
        {
            this.requestHandler = requestHandler ?? throw new ArgumentNullException(nameof(requestHandler));
        }

        /// <summary>
        /// ZMQ 리스너 시작
        /// </summary>
        public void Start(string address)
        {
            lock (lockObject)
            {
                if (isRunning)
                {
                    if (currentAddress != address)
                    {
                        RhinoApp.WriteLine($"ZMQ 리스너가 실행 중이지만 주소가 '{address}'(으)로 변경됨. 재시작 중...");
                        Stop();
                    }
                    else
                    {
                        return;
                    }
                }

                try
                {
                    socket = new ResponseSocket();

                    // 소켓 옵션 설정
                    socket.Options.Linger = TimeSpan.FromMilliseconds(500);
                    socket.Options.ReceiveHighWatermark = 1000;
                    socket.Options.SendHighWatermark = 1000;

                    RhinoApp.WriteLine($"ZMQ 연결 시도: {address}");
                    socket.Bind(address);

                    currentAddress = address;
                    isRunning = true;

                    // 리스닝 태스크 시작
                    cts = new CancellationTokenSource();
                    listenTask = Task.Factory.StartNew(ListenerLoop, cts.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);

                    RhinoApp.WriteLine($"ZMQ 리스너가 {address}에서 시작됨");
                }
                catch (Exception ex)
                {
                    RhinoApp.WriteLine($"ZMQ 리스너 시작 중 오류 발생: {ex.Message}");
                    Stop();
                }
            }
        }

        /// <summary>
        /// ZMQ 리스너 중지
        /// </summary>
        public void Stop()
        {
            lock (lockObject)
            {
                RhinoApp.WriteLine("ZMQ 리스너 중지 중...");

                // CTS 취소
                if (cts != null)
                {
                    try
                    {
                        cts.Cancel();
                        cts.Dispose();
                    }
                    catch (Exception ex)
                    {
                        RhinoApp.WriteLine($"CTS 취소 중 오류: {ex.Message}");
                    }
                    cts = null;
                }

                // 태스크 대기
                if (listenTask != null)
                {
                    try
                    {
                        listenTask.Wait(TimeSpan.FromMilliseconds(500));
                    }
                    catch (Exception ex)
                    {
                        RhinoApp.WriteLine($"태스크 대기 중 오류: {ex.Message}");
                    }
                    listenTask = null;
                }

                // 소켓 정리
                if (socket != null)
                {
                    try
                    {
                        if (!socket.IsDisposed)
                        {
                            socket.Unbind(currentAddress);
                            socket.Close();
                            socket.Dispose();
                        }
                    }
                    catch (Exception ex)
                    {
                        RhinoApp.WriteLine($"소켓 정리 중 오류: {ex.Message}");
                    }
                    socket = null;
                }

                isRunning = false;
                currentAddress = "";

                RhinoApp.WriteLine("ZMQ 리스너 중지 완료");
            }
        }

        /// <summary>
        /// 리스너 루프
        /// </summary>
        private void ListenerLoop()
        {
            string threadId = Thread.CurrentThread.ManagedThreadId.ToString();
            RhinoApp.WriteLine($"ZMQ 리스너 {threadId}: 루프 시작됨");

            try
            {
                while (!cts.IsCancellationRequested && socket != null && !socket.IsDisposed)
                {
                    // 요청 수신
                    string request = socket.ReceiveFrameString();

                    // 요청 처리
                    string response = "Error: Handler not available";
                    if (requestHandler != null)
                    {
                        try
                        {
                            response = requestHandler(request);
                        }
                        catch (Exception ex)
                        {
                            response = JsonConvert.SerializeObject(new { status = "error", message = $"요청 처리 중 오류: {ex.Message}" });
                        }
                    }

                    // 응답 전송
                    if (socket != null && !socket.IsDisposed)
                    {
                        socket.SendFrame(response);
                    }
                }
            }
            catch (TerminatingException)
            {
                RhinoApp.WriteLine("ZMQ 종료 요청으로 인해 리스너가 종료됨");
            }
            catch (ObjectDisposedException)
            {
                RhinoApp.WriteLine("소켓이 폐기되어 리스너가 종료됨");
            }
            catch (Exception ex)
            {
                RhinoApp.WriteLine($"ZMQ 리스너 루프 중 오류 발생: {ex.Message}");
            }

            RhinoApp.WriteLine($"ZMQ 리스너 {threadId}: 루프 종료됨");
        }

        /// <summary>
        /// IDisposable 구현
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed) return;

            if (disposing)
            {
                Stop();
            }

            disposed = true;
        }

        ~ZmqHandler()
        {
            Dispose(false);
        }
    }
}