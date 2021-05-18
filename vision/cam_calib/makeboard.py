import numpy as np
from PIL import Image
from cv2 import aruco

def makearucoboard(nrow, ncolumn, markerdict=aruco.DICT_6X6_250, startid = 0, markersize=25, savepath='./', name='test',
                   paperwidth=210, paperheight=297, dpi = 600, framesize = None):
    """
    create aruco board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction

    :param nrow:
    :param ncolumn:
    :param startid: the starting id of the marker
    :param markerdict:
    :param markersize:
    :param savepath:
    :param name: the name of the saved pdf file
    :param paperwidth: mm
    :param paperheight: mm
    :param dpi:
    :param framesize: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:

    author: weiwei
    date: 20190420
    """

    aruco_dict = aruco.Dictionary_get(markerdict)
    # 1mm = 0.0393701inch
    a4npxrow = int(paperheight*0.0393701*dpi)
    a4npxcolumn = int(paperwidth*0.0393701*dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255
    markersizepx = int(markersize*0.0393701*dpi)
    markerdist = int(markersizepx/4)

    if framesize is not None:
        framesize[0] = int(framesize[0]*0.0393701*dpi)
        framesize[1] = int(framesize[1]*0.0393701*dpi)
        if a4npxcolumn<framesize[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<framesize[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn-framesize[0])/2-1)
        framergt = int(framelft+1+framesize[0])
        frametop = int((a4npxrow-framesize[1])/2-1)
        framedown = int(frametop+1+framesize[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0

    markerareanpxrow = (nrow-1)*(markerdist)+nrow*markersizepx
    uppermargin = int((a4npxrow-markerareanpxrow)/2)
    markerareanpxcolumn = (ncolumn-1)*(markerdist)+ncolumn*markersizepx
    leftmargin = int((a4npxcolumn-markerareanpxcolumn)/2)

    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return

    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin+idnr*(markersizepx+markerdist)
            endrow = startrow+markersizepx
            startcolumn = leftmargin+idnc*(markersizepx+markerdist)
            endcolumn = markersizepx+startcolumn
            i = startid+idnr*ncolumn+idnc
            img = aruco.drawMarker(aruco_dict,i,markersizepx)
            bgimg[startrow:endrow, startcolumn:endcolumn] = img
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath+name+".pdf", "PDF", resolution=dpi)

def makecharucoboard(nrow, ncolumn, markerdict=aruco.DICT_4X4_250, squaresize=25, savepath='./',
                   paperwidth=210, paperheight=297, dpi = 600, framesize = None):
    """
    create charuco board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction

    :param nrow:
    :param ncolumn:
    :param markerdict:
    :param savepath:
    :param paperwidth: mm
    :param paperheight: mm
    :param dpi:
    :param framesize: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:

    author: weiwei
    date: 20190420
    """

    aruco_dict = aruco.Dictionary_get(markerdict)
    # 1mm = 0.0393701inch
    a4npxrow = int(paperheight*0.0393701*dpi)
    a4npxcolumn = int(paperwidth*0.0393701*dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255

    if framesize is not None:
        framesize[0] = int(framesize[0]*0.0393701*dpi)
        framesize[1] = int(framesize[1]*0.0393701*dpi)
        if a4npxcolumn<framesize[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<framesize[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn-framesize[0])/2-1)
        framergt = int(framelft+1+framesize[0])
        frametop = int((a4npxrow-framesize[1])/2-1)
        framedown = int(frametop+1+framesize[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0

    squaresizepx = int(squaresize*0.0393701*dpi)
    squareareanpxrow = squaresizepx*nrow
    uppermargin = int((a4npxrow-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx*ncolumn
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)

    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return

    board = aruco.CharucoBoard_create(ncolumn, nrow, squaresize, .57*squaresize, aruco_dict)
    imboard = board.draw((squareareanpxcolumn, squareareanpxrow))
    print(imboard.shape)
    startrow = uppermargin
    endrow = uppermargin+squareareanpxrow
    startcolumn = leftmargin
    endcolumn = leftmargin+squareareanpxcolumn
    bgimg[startrow:endrow, startcolumn:endcolumn] = imboard
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath+"test.pdf", "PDF", resolution=dpi)

def makechessboard(nrow, ncolumn, squaresize=25, savepath='./', paperwidth=210, paperheight=297, dpi = 600, framesize = None):
    """
    create checss board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction

    :param nrow:
    :param ncolumn:
    :param savepath:
    :param paperwidth: mm
    :param paperheight: mm
    :param dpi:
    :param framesize: [width, height] the 1pt frame for easy cut, nothing is drawn by default
    :return:

    author: weiwei
    date: 20190420
    """

    # 1mm = 0.0393701inch
    a4npxrow = int(paperheight*0.0393701*dpi)
    a4npxcolumn = int(paperwidth*0.0393701*dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255

    if framesize is not None:
        framesize[0] = int(framesize[0]*0.0393701*dpi)
        framesize[1] = int(framesize[1]*0.0393701*dpi)
        if a4npxcolumn<framesize[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<framesize[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn-framesize[0])/2-1)
        framergt = int(framelft+1+framesize[0])
        frametop = int((a4npxrow-framesize[1])/2-1)
        framedown = int(frametop+1+framesize[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0

    squaresizepx = int(squaresize*0.0393701*dpi)

    squareareanpxrow = squaresizepx*nrow
    uppermargin = int((a4npxrow-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx*ncolumn
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)

    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return

    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin+idnr*squaresizepx
            endrow = startrow+squaresizepx
            startcolumn = leftmargin+idnc*squaresizepx
            endcolumn = squaresizepx+startcolumn
            if idnr%2 != 0 and idnc%2 != 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
            if idnr%2 == 0 and idnc%2 == 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath+"test.pdf", "PDF", resolution=dpi)

    worldpoints = np.zeros((nrow*ncolumn, 3), np.float32)
    worldpoints[:, :2] = np.mgrid[:nrow, :ncolumn].T.reshape(-1, 2)*squaresize
    return worldpoints

def makechessandcharucoboard(nrowchess=3, ncolumnchess=5, squaresize=25,
                              nrowcharuco=3, ncolumncharuco=5, markerdict=aruco.DICT_6X6_250, squaresizearuco=25,
                              savepath='./', paperwidth=210, paperheight=297, dpi = 600, framesize = None):
    """
    create half-chess and half-charuco board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction

    :param nrow:
    :param ncolumn:
    :param squaresize: mm
    :param markerdict:
    :param savepath:
    :param paperwidth: mm
    :param paperheight: mm
    :param dpi:
    :param framesize: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:

    author: weiwei
    date: 20190420
    """

    aruco_dict = aruco.Dictionary_get(markerdict)
    # 1mm = 0.0393701inch
    a4npxrow = int(paperheight*0.0393701*dpi)
    a4npxcolumn = int(paperwidth*0.0393701*dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255

    if framesize is not None:
        framesize[0] = int(framesize[0]*0.0393701*dpi)
        framesize[1] = int(framesize[1]*0.0393701*dpi)
        if a4npxcolumn<framesize[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<framesize[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn-framesize[0])/2-1)
        framergt = int(framelft+1+framesize[0])
        frametop = int((a4npxrow-framesize[1])/2-1)
        framedown = int(frametop+1+framesize[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0

    # upper half, charuco
    squaresizepx = int(squaresizearuco*0.0393701*dpi)
    squareareanpxrow = squaresizepx*nrowchess
    uppermargin = int((a4npxrow/2-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx*ncolumnchess
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)

    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return

    board = aruco.CharucoBoard_create(ncolumnchess, nrowchess, squaresizearuco, .57*squaresizearuco, aruco_dict)
    imboard = board.draw((squareareanpxcolumn, squareareanpxrow))
    print(imboard.shape)
    startrow = uppermargin
    endrow = uppermargin+squareareanpxrow
    startcolumn = leftmargin
    endcolumn = leftmargin+squareareanpxcolumn
    bgimg[startrow:endrow, startcolumn:endcolumn] = imboard

    # lower half, chess
    squaresizepx = int(squaresize*0.0393701*dpi)

    squareareanpxrow = squaresizepx*nrowcharuco
    uppermargin = int((a4npxrow/2-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx*ncolumncharuco
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)

    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return

    for idnr in range(nrowcharuco):
        for idnc in range(ncolumncharuco):
            startrow = int(a4npxrow/2)+uppermargin+idnr*squaresizepx
            endrow = startrow+squaresizepx
            startcolumn = leftmargin+idnc*squaresizepx
            endcolumn = squaresizepx+startcolumn
            if idnr%2 != 0 and idnc%2 != 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
            if idnr%2 == 0 and idnc%2 == 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0

    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath+"test.pdf", "PDF", resolution=dpi)

if __name__ == '__main__':
    # makechessandcharucoboard(4,6,32,5,7)
    # makecharucoboard(7,5, square_size=40)
    # makechessboard(7,5, square_size=40)
    # makearucoboard(2,2, marker_size=80)
    # makearucoboard(1,1,marker_dict=aruco.DICT_4X4_250, start_id=1, marker_size=45, frame_size=[60,60])

    makechessboard(1, 1, squaresize=35, framesize = [100,150])