#include "libmotioncapture/vicon.h"

// VICON
#include "ViconDataStreamSDK_CPP/DataStreamClient.h"

using namespace ViconDataStreamSDK::CPP;

namespace libmotioncapture {

  class MotionCaptureViconImpl
  {
  public:
    Client client;
    std::string version;
  };

  MotionCaptureVicon::MotionCaptureVicon(
    const std::string& hostname,
    bool enableObjects,
    bool enablePointcloud)
  {
    pImpl = new MotionCaptureViconImpl;

    // Try connecting...
    while (!pImpl->client.IsConnected().Connected) {
      pImpl->client.Connect(hostname);
    }

    if (enableObjects) {
      pImpl->client.EnableSegmentData();
    }
    if (enablePointcloud) {
      pImpl->client.EnableUnlabeledMarkerData();
    }

    // This is the lowest latency option
    pImpl->client.SetStreamMode(ViconDataStreamSDK::CPP::StreamMode::ServerPush);

    // Set the global up axis
    pImpl->client.SetAxisMapping(Direction::Forward,
                          Direction::Left,
                          Direction::Up); // Z-up

    // Discover the version number
    Output_GetVersion version = pImpl->client.GetVersion();
    std::stringstream sstr;
    sstr << version.Major << "." << version.Minor << "." << version.Point;
    pImpl->version = sstr.str();
  }

  MotionCaptureVicon::~MotionCaptureVicon()
  {
    delete pImpl;
  }

  const std::string& MotionCaptureVicon::version() const
  {
    return pImpl->version;
  }

  void MotionCaptureVicon::waitForNextFrame()
  {
    while (pImpl->client.GetFrame().Result != Result::Success) {
    }
  }

  const std::map<std::string, RigidBody>& MotionCaptureVicon::rigidBodies() const
  {
    rigidBodies_.clear();
    size_t count = pImpl->client.GetSubjectCount().SubjectCount;
    for (size_t i = 0; i < count; ++i) {
      const std::string name = pImpl->client.GetSubjectName(i).SubjectName;
      rigidBodies_.emplace(name, rigidBodyByName(name));
    }
    return rigidBodies_;
  }

  RigidBody MotionCaptureVicon::rigidBodyByName(
    const std::string& name) const
  {
    auto const translation = pImpl->client.GetSegmentGlobalTranslation(name, name);
    auto const quaternion = pImpl->client.GetSegmentGlobalRotationQuaternion(name, name);
    if (   translation.Result == Result::Success
        && quaternion.Result == Result::Success
        && !translation.Occluded
        && !quaternion.Occluded) {

      Eigen::Vector3f position(
        translation.Translation[0] / 1000.0,
        translation.Translation[1] / 1000.0,
        translation.Translation[2] / 1000.0);

      Eigen::Quaternionf rotation(
        quaternion.Rotation[3], // w
        quaternion.Rotation[0], // x
        quaternion.Rotation[1], // y
        quaternion.Rotation[2]  // z
        );

      return RigidBody(name, position, rotation);
    }
    // [dsl-2022.08.29] For safety, we shouldn't throw a runtime_error when rigid body is lost
    // throw std::runtime_error("Unknown rigid body!");
    printf("Unknown rigid body!\n");

    // return dummy rigid body 
    Eigen::Vector3f dummy_pos(
      0.0,
      0.0,
      0.0);

    Eigen::Quaternionf dummy_rot(
      1.0,   
      0.0,
      0.0,
      0.0
    );

    return RigidBody("dummy_name", dummy_pos, dummy_rot);
  }

  const PointCloud& MotionCaptureVicon::pointCloud() const
  {
    size_t count = pImpl->client.GetUnlabeledMarkerCount().MarkerCount;
    pointcloud_.resize(count, Eigen::NoChange);
    for(size_t i = 0; i < count; ++i) {
      Output_GetUnlabeledMarkerGlobalTranslation translation =
        pImpl->client.GetUnlabeledMarkerGlobalTranslation(i);
      pointcloud_.row(i) << 
        translation.Translation[0] / 1000.0,
        translation.Translation[1] / 1000.0,
        translation.Translation[2] / 1000.0;
    }
    return pointcloud_;
  }

  const std::vector<LatencyInfo>& MotionCaptureVicon::latency() const
  {
    latencies_.clear();
    size_t latencyCount = pImpl->client.GetLatencySampleCount().Count;
    for(size_t i = 0; i < latencyCount; ++i) {
      std::string sampleName  = pImpl->client.GetLatencySampleName(i).Name;
      double      sampleValue = pImpl->client.GetLatencySampleValue(sampleName).Value;
      latencies_.emplace_back(LatencyInfo(sampleName, sampleValue));
    }
    return latencies_;
  }

}
