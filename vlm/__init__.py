"""Publishing this bench's measurements into the VLM framework.

The framework (`Mobile-manipulation-with-VLMs`) is a ZMQ-RPC service mesh whose
agent layer, planner and UI all reach the world through one boundary:
`DataService` for what is seen and `RobotService` for what is commanded. This
package implements that boundary against a Sphero and a blob tracker, so
nothing above it has to know the robot is a ball.

Nothing in here is authoritative. The bench measures and drives; this only
carries the result across.
"""
