from wrs import drivers as xai
import math

xai_x = xai.XArmAPI(port="192.168.1.227")
if xai_x.has_err_warn:
    if xai_x.get_err_warn_code()[1][0] == 1:
        print("The Emergency Button is pushed in to stop!")
        input("Release the emergency button and press any key to continue. Press Enter to continue...")
    xai_x.clean_error()
    xai_x.clean_error()

xai_x.motion_enable()
xai_x.set_mode(1) # servo motion mode
xai_x.set_state(state=0)
xai_x.reset(wait=True)

angles = [tg for tg in xai_x.get_servo_angle(is_radian=True)[1]]
print(angles)
angles[1]+=.3
print(angles)
print(xai_x.set_servo_angle_j(angles=angles, speed = math.radians(10), is_radian=True))