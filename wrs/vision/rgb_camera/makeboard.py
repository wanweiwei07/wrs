import numpy as np
from PIL import Image
from cv2 import aruco


def make_arucoboard(n_row, n_column, marker_dict=aruco.DICT_6X6_250, start_id=0, marker_size=25, save_path='./',
                    name='test', paper_width=210, paper_height=297, dpi=600, frame_size=None):
    """
    create aruco board
    the paper is in portrait orientation, n_row means the number of markers in the vertical motion_vec
    :param n_row:
    :param n_column:
    :param start_id: the starting id of the marker
    :param marker_dict:
    :param marker_size:
    :param save_path:
    :param name: the name of the saved pdf file
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :param frame_size: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:
    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.Dictionary_get(marker_dict)
    # 1mm = 0.0393701inch
    a4_n_pxrow = int(paper_height * 0.0393701 * dpi)
    a4_n_pxcolumn = int(paper_width * 0.0393701 * dpi)
    bg_img = np.ones((a4_n_pxrow, a4_n_pxcolumn), dtype='uint8') * 255
    marker_size_in_px = int(marker_size * 0.0393701 * dpi)
    marker_dist_in_px = int(marker_size_in_px / 4)
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * 0.0393701 * dpi)
        frame_size[1] = int(frame_size[1] * 0.0393701 * dpi)
        if a4_n_pxcolumn < frame_size[0] + 2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4_n_pxrow < frame_size[1] + 2:
            print("Frame height must be smaller than the #pt in each column.")
        frame_lft = int((a4_n_pxcolumn - frame_size[0]) / 2 - 1)
        frame_rgt = int(frame_lft + 1 + frame_size[0])
        frame_top = int((a4_n_pxrow - frame_size[1]) / 2 - 1)
        frame_down = int(frame_top + 1 + frame_size[1])
        bg_img[frame_top:frame_down + 1, frame_lft:frame_lft + 1] = 0
        bg_img[frame_top:frame_down + 1, frame_rgt:frame_rgt + 1] = 0
        bg_img[frame_top:frame_top + 1, frame_lft:frame_rgt + 1] = 0
        bg_img[frame_down:frame_down + 1, frame_lft:frame_rgt + 1] = 0
    marker_area_n_pxrow = (n_row - 1) * (marker_dist_in_px) + n_row * marker_size_in_px
    upper_margin = int((a4_n_pxrow - marker_area_n_pxrow) / 2)
    marker_area_n_pxcolumn = (n_column - 1) * (marker_dist_in_px) + n_column * marker_size_in_px
    left_margin = int((a4_n_pxcolumn - marker_area_n_pxcolumn) / 2)
    if (upper_margin <= 10) or (left_margin <= 10):
        print("Too many markers! Reduce n_row and n_column.")
        return
    for id_row in range(n_row):
        for id_column in range(n_column):
            start_row = upper_margin + id_row * (marker_size_in_px + marker_dist_in_px)
            end_row = start_row + marker_size_in_px
            start_column = left_margin + id_column * (marker_size_in_px + marker_dist_in_px)
            end_column = marker_size_in_px + start_column
            i = start_id + id_row * n_column + id_column
            img = aruco.drawMarker(aruco_dict, i, marker_size_in_px)
            bg_img[start_row:end_row, start_column:end_column] = img
    im = Image.fromarray(bg_img).convert("L")
    im.save(save_path + name + ".pdf", "PDF", resolution=dpi)


def make_charucoboard(n_row, n_column, marker_dict=aruco.DICT_4X4_250, square_size=25, save_path='./',
                      paper_width=210, paper_height=297, dpi=600, frame_size=None):
    """
    create charuco board
    the paper is in portrait orientation, n_row means the number of markers in the vertical motion_vec
    :param n_row:
    :param n_column:
    :param marker_dict:
    :param save_path:
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :param frame_size: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:
    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.Dictionary_get(marker_dict)
    # 1mm = 0.0393701inch
    a4_n_pxrow = int(paper_height * 0.0393701 * dpi)
    a4_n_pxcolumn = int(paper_width * 0.0393701 * dpi)
    bg_img = np.ones((a4_n_pxrow, a4_n_pxcolumn), dtype='uint8') * 255
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * 0.0393701 * dpi)
        frame_size[1] = int(frame_size[1] * 0.0393701 * dpi)
        if a4_n_pxcolumn < frame_size[0] + 2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4_n_pxrow < frame_size[1] + 2:
            print("Frame height must be smaller than the #pt in each column.")
        frame_lft = int((a4_n_pxcolumn - frame_size[0]) / 2 - 1)
        frame_rgt = int(frame_lft + 1 + frame_size[0])
        frame_top = int((a4_n_pxrow - frame_size[1]) / 2 - 1)
        frame_down = int(frame_top + 1 + frame_size[1])
        bg_img[frame_top:frame_down + 1, frame_lft:frame_lft + 1] = 0
        bg_img[frame_top:frame_down + 1, frame_rgt:frame_rgt + 1] = 0
        bg_img[frame_top:frame_top + 1, frame_lft:frame_rgt + 1] = 0
        bg_img[frame_down:frame_down + 1, frame_lft:frame_rgt + 1] = 0
    square_size_in_px = int(square_size * 0.0393701 * dpi)
    square_area_n_pxrow = square_size_in_px * n_row
    upper_margin = int((a4_n_pxrow - square_area_n_pxrow) / 2)
    square_area_n_pxcolumn = square_size_in_px * n_column
    left_margin = int((a4_n_pxcolumn - square_area_n_pxcolumn) / 2)
    if (upper_margin <= 10) or (left_margin <= 10):
        print("Too many markers! Reduce n_row and n_column.")
        return
    board = aruco.CharucoBoard_create(n_column, n_row, square_size, .57 * square_size, aruco_dict)
    imboard = board.draw((square_area_n_pxcolumn, square_area_n_pxrow))
    print(imboard.shape)
    start_row = upper_margin
    end_row = upper_margin + square_area_n_pxrow
    start_column = left_margin
    end_column = left_margin + square_area_n_pxcolumn
    bg_img[start_row:end_row, start_column:end_column] = imboard
    im = Image.fromarray(bg_img).convert("L")
    im.save(save_path + "test.pdf", "PDF", resolution=dpi)


def make_chessboard(n_row, n_column, square_size=25, save_path='./', paper_width=210, paper_height=297, dpi=600,
                    frame_size=None):
    """
    create checss board
    the paper is in portrait orientation, n_row means the number of markers in the vertical motion_vec

    :param n_row:
    :param n_column:
    :param save_path:
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :param frame_size: [width, height] the 1pt frame for easy cut, nothing is drawn by default
    :return:

    author: weiwei
    date: 20190420
    """

    # 1mm = 0.0393701inch
    a4_n_pxrow = int(paper_height * 0.0393701 * dpi)
    a4_n_pxcolumn = int(paper_width * 0.0393701 * dpi)
    bg_img = np.ones((a4_n_pxrow, a4_n_pxcolumn), dtype='uint8') * 255

    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * 0.0393701 * dpi)
        frame_size[1] = int(frame_size[1] * 0.0393701 * dpi)
        if a4_n_pxcolumn < frame_size[0] + 2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4_n_pxrow < frame_size[1] + 2:
            print("Frame height must be smaller than the #pt in each column.")
        frame_lft = int((a4_n_pxcolumn - frame_size[0]) / 2 - 1)
        frame_rgt = int(frame_lft + 1 + frame_size[0])
        frame_top = int((a4_n_pxrow - frame_size[1]) / 2 - 1)
        frame_down = int(frame_top + 1 + frame_size[1])
        bg_img[frame_top:frame_down + 1, frame_lft:frame_lft + 1] = 0
        bg_img[frame_top:frame_down + 1, frame_rgt:frame_rgt + 1] = 0
        bg_img[frame_top:frame_top + 1, frame_lft:frame_rgt + 1] = 0
        bg_img[frame_down:frame_down + 1, frame_lft:frame_rgt + 1] = 0

    square_size_in_px = int(square_size * 0.0393701 * dpi)

    square_area_n_pxrow = square_size_in_px * n_row
    upper_margin = int((a4_n_pxrow - square_area_n_pxrow) / 2)
    square_area_n_pxcolumn = square_size_in_px * n_column
    left_margin = int((a4_n_pxcolumn - square_area_n_pxcolumn) / 2)

    if (upper_margin <= 10) or (left_margin <= 10):
        print("Too many markers! Reduce n_row and n_column.")
        return

    for id_row in range(n_row):
        for id_column in range(n_column):
            start_row = upper_margin + id_row * square_size_in_px
            end_row = start_row + square_size_in_px
            start_column = left_margin + id_column * square_size_in_px
            end_column = square_size_in_px + start_column
            if id_row % 2 != 0 and id_column % 2 != 0:
                bg_img[start_row:end_row, start_column:end_column] = 0
            if id_row % 2 == 0 and id_column % 2 == 0:
                bg_img[start_row:end_row, start_column:end_column] = 0
    im = Image.fromarray(bg_img).convert("L")
    im.save(save_path + "test.pdf", "PDF", resolution=dpi)

    world_points = np.zeros((n_row * n_column, 3), np.float32)
    world_points[:, :2] = np.mgrid[:n_row, :n_column].T.reshape(-1, 2) * square_size
    return world_points


def make_chess_and_charucoboard(n_row_chess=3, n_column_chess=5, square_size=25,
                                n_row_charuco=3, n_column_charuco=5, marker_dict=aruco.DICT_6X6_250,
                                square_size_aruco=25, save_path='./', paper_width=210, paper_height=297, dpi=600,
                                frame_size=None):
    """
    create half-chess and half-charuco board
    the paper is in portrait orientation, n_row means the number of markers in the vertical motion_vec

    :param n_row:
    :param n_column:
    :param square_size: mm
    :param marker_dict:
    :param save_path:
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :param frame_size: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:

    author: weiwei
    date: 20190420
    """

    aruco_dict = aruco.Dictionary_get(marker_dict)
    # 1mm = 0.0393701inch
    a4_n_pxrow = int(paper_height * 0.0393701 * dpi)
    a4_n_pxcolumn = int(paper_width * 0.0393701 * dpi)
    bg_img = np.ones((a4_n_pxrow, a4_n_pxcolumn), dtype='uint8') * 255

    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * 0.0393701 * dpi)
        frame_size[1] = int(frame_size[1] * 0.0393701 * dpi)
        if a4_n_pxcolumn < frame_size[0] + 2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4_n_pxrow < frame_size[1] + 2:
            print("Frame height must be smaller than the #pt in each column.")
        frame_lft = int((a4_n_pxcolumn - frame_size[0]) / 2 - 1)
        frame_rgt = int(frame_lft + 1 + frame_size[0])
        frame_top = int((a4_n_pxrow - frame_size[1]) / 2 - 1)
        frame_down = int(frame_top + 1 + frame_size[1])
        bg_img[frame_top:frame_down + 1, frame_lft:frame_lft + 1] = 0
        bg_img[frame_top:frame_down + 1, frame_rgt:frame_rgt + 1] = 0
        bg_img[frame_top:frame_top + 1, frame_lft:frame_rgt + 1] = 0
        bg_img[frame_down:frame_down + 1, frame_lft:frame_rgt + 1] = 0

    # upper half, charuco
    square_size_in_px = int(square_size_aruco * 0.0393701 * dpi)
    square_area_n_pxrow = square_size_in_px * n_row_chess
    upper_margin = int((a4_n_pxrow / 2 - square_area_n_pxrow) / 2)
    square_area_n_pxcolumn = square_size_in_px * n_column_chess
    left_margin = int((a4_n_pxcolumn - square_area_n_pxcolumn) / 2)

    if (upper_margin <= 10) or (left_margin <= 10):
        print("Too many markers! Reduce n_row and n_column.")
        return

    board = aruco.CharucoBoard_create(n_column_chess, n_row_chess, square_size_aruco, .57 * square_size_aruco,
                                      aruco_dict)
    imboard = board.draw((square_area_n_pxcolumn, square_area_n_pxrow))
    print(imboard.shape)
    start_row = upper_margin
    end_row = upper_margin + square_area_n_pxrow
    start_column = left_margin
    end_column = left_margin + square_area_n_pxcolumn
    bg_img[start_row:end_row, start_column:end_column] = imboard

    # lower half, chess
    square_size_in_px = int(square_size * 0.0393701 * dpi)

    square_area_n_pxrow = square_size_in_px * n_row_charuco
    upper_margin = int((a4_n_pxrow / 2 - square_area_n_pxrow) / 2)
    square_area_n_pxcolumn = square_size_in_px * n_column_charuco
    left_margin = int((a4_n_pxcolumn - square_area_n_pxcolumn) / 2)

    if (upper_margin <= 10) or (left_margin <= 10):
        print("Too many markers! Reduce n_row and n_column.")
        return

    for id_row in range(n_row_charuco):
        for id_column in range(n_column_charuco):
            start_row = int(a4_n_pxrow / 2) + upper_margin + id_row * square_size_in_px
            end_row = start_row + square_size_in_px
            start_column = left_margin + id_column * square_size_in_px
            end_column = square_size_in_px + start_column
            if id_row % 2 != 0 and id_column % 2 != 0:
                bg_img[start_row:end_row, start_column:end_column] = 0
            if id_row % 2 == 0 and id_column % 2 == 0:
                bg_img[start_row:end_row, start_column:end_column] = 0

    im = Image.fromarray(bg_img).convert("L")
    im.save(save_path + "test.pdf", "PDF", resolution=dpi)


if __name__ == '__main__':
    # make_chess_and_charucoboard(4,6,32,5,7)
    # make_charucoboard(7,5, square_size=40)
    # make_chessboard(7,5, square_size=40)
    # make_arucoboard(2,2, marker_size=80)
    # make_arucoboard(1,1,marker_dict=aruco.DICT_4X4_250, start_id=1, marker_size=45, frame_size=[60,60])

    make_chessboard(1, 1, square_size=35, frame_size=[100, 150])
