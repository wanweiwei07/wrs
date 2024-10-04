import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wrs.vision.coded_aperture.code_aperture import mura

result = mura(rank=5)
result.plot()
plt.show()

_MM_TO_INCH = 0.03937008


def make_mura(rank=5,
              code_size=5,  # approximate this value while rounding floating values
              frame_size=(20, 20),
              name='test',
              paper_width=210,
              paper_height=297,
              dpi=1200,
              option='pdf'):
    """
    :param rank: 0->5, 1->13, 2->17, 3->29, 4-> 37, 5->41, 6->53, 7->61, 8->73, 9->89, 10->97, etc. prime(4*(id-1)+1)
    :param code_size: mm
    :param save_path:
    :param name:
    :param paper_width:
    :param paper_height:
    :param dpi:
    :return:
    """
    # coded aperture
    code = mura(rank=rank)
    aperture = code.aperture.T
    mura_length = len(aperture)
    mm_per_hole = code_size / mura_length
    hole_dot = int(mm_per_hole * _MM_TO_INCH * dpi)
    aperture_dot = hole_dot * mura_length
    # paper array
    paper_n_row = int(paper_height * _MM_TO_INCH * dpi)
    paper_n_column = int(paper_width * _MM_TO_INCH * dpi)
    paper_array = np.ones((paper_n_row, paper_n_column), dtype='uint8') * 255
    # bg
    bg_n_row = int(frame_size[0] * _MM_TO_INCH * dpi)
    bg_n_column = int(frame_size[1] * _MM_TO_INCH * dpi)
    lft = int((paper_n_column - bg_n_column) / 2)
    rgt = lft + bg_n_column
    top = int((paper_n_row - bg_n_row) / 2)
    down = top + bg_n_row
    paper_array[top:down, lft:rgt] = 0
    # aperture dots
    aperture_lft = int((paper_n_column - aperture_dot) / 2 - 1)
    aperture_top = int((paper_n_row - aperture_dot) / 2 - 1)
    # fill the coded aperture
    for i in range(mura_length):
        for j in range(mura_length):
            hole_lft = int(aperture_lft + 1 + j * hole_dot)
            hole_right = int(hole_lft + hole_dot)
            hole_top = int(aperture_top + 1 + i * hole_dot)
            hole_down = int(hole_top + hole_dot)
            if aperture[i, j] == 1:
                paper_array[hole_top:hole_down, hole_lft:hole_right] = 255
    if option == 'pdf':
        im = Image.fromarray(paper_array).convert("L")
        im.save(name + ".pdf", "PDF", resolution=dpi)
    elif option == 'return':
        return paper_array
    else:
        raise ValueError("option must be 'pdf' or 'return'!")


def make_mura_at_given_pos(rank=5,
                           code_size=5,  # approximate this value while rounding floating values
                           frame_size=(25, 25),
                           position=(20, 20),
                           name='test',
                           paper_width=210,
                           paper_height=297,
                           dpi=1200,
                           option='pdf'):
    """
    :param rank:
    :param code_size:
    :param frame_size: mm on paper
    :param position: mm on paper
    :param name:
    :param paper_width:
    :param paper_height:
    :param dpi:
    :param option:
    :return:
    """
    # coded aperture
    code = mura(rank=rank)
    aperture = code.aperture.T
    mura_length = len(aperture)
    mm_per_hole = code_size / mura_length
    hole_dot = int(mm_per_hole * _MM_TO_INCH * dpi)
    aperture_dot = hole_dot * mura_length
    aperture_mm = aperture_dot / dpi / _MM_TO_INCH
    # paper array
    paper_n_row = int(paper_height * _MM_TO_INCH * dpi)
    paper_n_column = int(paper_width * _MM_TO_INCH * dpi)
    paper_array = np.ones((paper_n_row, paper_n_column), dtype='uint8') * 255
    # position
    position_row = int(position[0] * _MM_TO_INCH * dpi)
    position_column = int(position[1] * _MM_TO_INCH * dpi)
    # bg
    bg_n_row = int(frame_size[0] * _MM_TO_INCH * dpi)
    bg_n_column = int(frame_size[1] * _MM_TO_INCH * dpi)
    lft = position_column - int(bg_n_column / 2)
    rgt = lft + bg_n_column
    top = position_row - int(bg_n_row / 2)
    down = top + bg_n_row
    paper_array[top:down, lft:rgt] = 0
    # aperture dots
    aperture_lft = position_column - int(aperture_dot / 2 - 1)
    aperture_top = position_row - int(aperture_dot / 2 - 1)
    # fill the coded aperture
    for i in range(mura_length):
        for j in range(mura_length):
            hole_lft = int(aperture_lft + 1 + j * hole_dot)
            hole_right = int(hole_lft + hole_dot)
            hole_top = int(aperture_top + 1 + i * hole_dot)
            hole_down = int(hole_top + hole_dot)
            if aperture[i, j] == 1:
                paper_array[hole_top:hole_down, hole_lft:hole_right] = 255
    if option == 'pdf':
        im = Image.fromarray(paper_array).convert("L")
        im.save(name + ".pdf", "PDF", resolution=dpi)
    elif option == 'return':
        return paper_array
    else:
        raise ValueError("option must be 'pdf' or 'return'!")


def make_multi_code(rank=5,
                    code_size=5,
                    frame_size=(25, 25),
                    name='test',
                    paper_width=210,
                    paper_height=297,
                    dpi=1200,
                    option='pdf'):
    """
    make multiples on a single page
    :param rank:
    :param code_size:
    :param frame_size:
    :param name:
    :param paper_width:
    :param paper_height:
    :param dpi:
    :param option:
    :return:
    author: weiwei
    date: 20240707
    """
    # paper array
    paper_n_row = int(paper_height * _MM_TO_INCH * dpi)
    paper_n_column = int(paper_width * _MM_TO_INCH * dpi)
    paper_array = np.ones((paper_n_row, paper_n_column), dtype='uint8') * 255
    # content
    row = frame_size[0] / 2 + 7.5
    column = frame_size[1] / 2 + 7.5
    row_span = frame_size[0] * 2
    column_span = frame_size[1] * 2
    while True:
        if row > paper_height:
            break
        while True:
            if column > paper_width:
                break
            temp_array = make_mura_at_given_pos(rank=rank,
                                                code_size=code_size,
                                                frame_size=frame_size,
                                                position=(row, column),
                                                name=name,
                                                paper_width=paper_width,
                                                paper_height=paper_height,
                                                dpi=dpi,
                                                option='return')
            paper_array = np.where(temp_array < 255, temp_array, paper_array)
            column += column_span
        row += row_span
        column = frame_size[1] / 2 + 7.5
    if option == 'pdf':
        im = Image.fromarray(paper_array).convert("L")
        im.save(name + ".pdf", "PDF", resolution=dpi)
    elif option == 'return':
        return paper_array
    else:
        raise ValueError("option must be 'pdf' or 'return'!")


if __name__ == '__main__':
    make_multi_code(rank=5)
