cmake_minimum_required(VERSION 2.8.3)
project(merganzer_line_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  duckietown_msgs # Every duckietown packages should use this.
  cv_bridge
  # XXX add anti_instagram?
)

add_message_files(
  FILES
  BezierMsg.msg
  BeziersMsg.msg
  SkeletonMsg.msg
  SkeletonsMsg.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
