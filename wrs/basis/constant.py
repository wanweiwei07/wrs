import numpy as np
import matplotlib.pyplot as plt

eps = 1e-6
pi = 3.14159265359

# color
# see correspondence at: https://www.colordic.org/w
red = np.array([1, 0, 0, 1])
green = np.array((0.0, 0.5019607843137255, 0.0, 1.0))
blue = np.array([0, 0, 1, 1])
cyan = np.array([0, 1, 1, 1])
magenta = np.array([1, 0, 1, 1])
yellow = np.array([1, 1, 0, 1])
black = np.array([0, 0, 0, 1])
orange = np.array([1,0.47,0,1])
white = np.array([1, 1, 1, 1])
gray=np.array([0.67, .67, .67, 1])
chinese_red = np.array([0.9294, 0.4275, 0.2745, 1])  # 237, 109, 70
carrot_orange = np.array([0.9294, 0.4275, 0.2078, 1])  # 237, 109, 53
chocolate = np.array([0.4235, 0.2078, 0.1412, 1])  # 108, 53, 36
salmon_pink = np.array([0.9529, 0.6510, 0.5490, 1])  # 243, 166, 140
antique_gold = np.array([.7569, 0.6706, 0.0196, 1])  # 193, 171, 5
olive = np.array([0.4471, 0.3922, 0.0471, 1])  # 114, 100, 12

ivory = np.array([0.9725, 0.9569, 0.9020, 1])  # 248, 244, 230
beige = np.array([0.961, 0.961, 0.863, 1.0]) # 245, 245, 220

violet = np.array([0.3529, 0.2667, 0.5961, 1])  # 90, 68, 152
royal_purple = np.array([0.4980, 0.0667, 0.5176, 1])  # 127, 17, 132
moon_gray = np.array([0.8314, 0.8510, 0.8627, 1])  # 212, 217, 220
china_clay = np.array([0.8314, 0.8627, 0.8275, 1])  # 212, 220, 211
silver_gray = np.array([0.6863, 0.6863, 0.6902, 1])  # 175, 175, 176
navy_blue = np.array([0.1255, 0.1843, 0.3333, 1])  # 32, 47, 85
oriental_blue = np.array([0.1490, 0.2863, 0.6157, 1])  # 38, 73, 157
# css color (see the css_color_table_picture)
light_coral = np.array((0.9411764705882353, 0.5019607843137255, 0.5019607843137255, 1.0))
orange_red = np.array((1.0, 0.27058823529411763, 0.0, 1.0))

tomato = np.array((1.0, 0.38823529411764707, 0.2784313725490196, 1.0))
lawn_green = np.array((0.48627450980392156, 0.9882352941176471, 0.0, 1.0))
deep_sky_blue = np.array((0.0, 0.7490196078431373, 1.0, 1.0))

deep_pink = np.array((1.0, 0.0784313725490196, 0.5764705882352941, 1.0))
pink = np.array((1.0, 0.7529411764705882, 0.796078431372549, 1.0))
spring_green = np.array((0.0, 1.0, 0.4980392156862745, 1.0))

# steel colors
steel_blue = np.array((0.274, 0.510, 0.706, 1.0)) # 70, 130, 180
light_steel_blue = np.array([0.690, 0.769, 0.871, 1.0]) # 176, 196, 222
steel_gray = np.array([0.4510, 0.4275, 0.4431, 1])  # 115, 109, 113
slate_gray = np.array([0.439, 0.502, 0.565, 1.0]) # 112, 128, 144
light_slate_gray = np.array([0.467, 0.533, 0.600, 1.0]) # 119, 136, 153
dim_gray = np.array([0.412, 0.412, 0.412, 1.0]) # 105, 105, 105
silver = np.array([0.753, 0.753, 0.753, 1.0]) # 192, 192, 192

lime = np.array([0, 1, 0, 1])
gold = np.array((1.0, 0.8431372549019608, 0.0, 1.0))
yellow_green = np.array((0.6039215686274509, 0.803921568627451, 0.19607843137254902, 1.0))
# default values
jnt_child_rgba = silver_gray
jnt_parent_rgba = steel_gray
lnk_stick_rgba = light_steel_blue
# default color mats
rgb_mat = np.column_stack((red[:3], green[:3], blue[:3]))
myc_mat = np.column_stack((magenta[:3], yellow[:3], cyan[:3]))
tld_mat = np.column_stack((tomato[:3], lawn_green[:3], deep_sky_blue[:3]))  # carrot orange, olive, navy blue
dyo_mat = np.column_stack((deep_pink[:3], yellow_green[:3], oriental_blue[:3]))  # carrot orange, olive, navy blue

x_ax = np.array([1, 0, 0])
y_ax = np.array([0, 1, 0])
z_ax = np.array([0, 0, 1])

# matplotlib colors
plasma_map = plt.get_cmap("plasma")
cividis_map = plt.get_cmap("cividis")
jet_map = plt.get_cmap("jet")
winter_map = plt.get_cmap("winter")
summer_map = plt.get_cmap("summer")
cool_map = plt.get_cmap("cool")
wistia_map = plt.get_cmap("Wistia")
spring_map = plt.get_cmap("spring")
copper_map = plt.get_cmap("copper")

tab20_list = [(r,g,b,1.0) for r,g,b in plt.get_cmap("tab20").colors]
hug_blue = tab20_list[0]
hug_gray = tab20_list[15]
# rgb_mat = np.column_stack((tab20_list[6][:3], tab20_list[4][:3], tab20_list[0][:3]))

if __name__ == '__main__':
    # def convert_mcolor_to_rgba(mcolor_name):
    #     print(f"np.array({mcolors.to_rgba(mcolor_name)})")
    #
    #
    # convert_mcolor_to_rgba("yellowgreen")

    # plot colors

    import matplotlib.pyplot as plt
    # Create a figure and axis
    fig, ax = plt.subplots()

    colors = {"red": red}
    # Plot each color as a rectangle
    for i, (name, color) in enumerate(colors.items()):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    # Set the x-axis tick labels
    plt.xticks(range(len(colors)), list(colors.keys()), rotation=90)

    # Set the limits of the plot
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)

    # Hide the axes
    ax.axis('off')

    # Show the plot
    plt.show()
