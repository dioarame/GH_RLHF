using System;
using System.Drawing; // Ensure this namespace is included
using System.Drawing.Imaging; // Add this if needed for Bitmap-related operations
using Grasshopper.Kernel;

namespace GrasshopperZmqComponent
{
    public class GrasshopperZmqInfo : GH_AssemblyInfo
    {
        public override string Name => "ZeroMQ Component";
        public override Bitmap Icon => null; // This line is fine if no icon is needed
        public override string Description => "ZeroMQ communication components for Grasshopper";
        public override Guid Id => new Guid("B4E9B1D7-145F-4A55-96F5-1FD5F3D7B965");
        public override string AuthorName => "Yang Seung-Won";
        public override string AuthorContact => "valenpoin@outlook.kr";
    }
}
