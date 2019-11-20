cmake_minimum_required(VERSION 2.8.3)
project(merganser_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  duckietown_msgs # Every duckietown packages should use this.
  message_generation
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
   duckietown_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
