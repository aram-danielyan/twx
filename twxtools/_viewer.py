"""
Copyright (C) 2022 by Aram Danielyan

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

@author: Aram Danielyan
"""
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons, Widget
from matplotlib import rcParams, rcParamsDefault, get_backend, _pylab_helpers
from matplotlib.widgets import RectangleSelector
import functools
import logging

def bring_to_front(fig):
    """
    Bring a Matplotlib figure to front.
    Supports MacOSX backend (via AppKit if available), TkAgg, and Qt backends.
    Silently does nothing if the backend is unsupported or required libs are missing.
    """
    fig.canvas.draw()
    backend = get_backend().lower()

    if 'macosx' in backend:
        try:
            import importlib
            AppKit = importlib.import_module('AppKit')
            app = AppKit.NSApplication.sharedApplication()
            app.activateIgnoringOtherApps_(True)
        except ImportError:
            pass
    elif 'tk' in backend:
        try:
            win = fig.canvas.manager.window
            win.lift()
            win.attributes('-topmost', True)
            win.after(200, lambda: win.attributes('-topmost', False))
        except Exception:
            pass
    elif 'qt' in backend:
        try:
            win = fig.canvas.manager.window
            win.raise_()
            win.activateWindow()
        except Exception:
            pass
    else:
        try:
            fig.canvas.manager.window.raise_()
        except Exception:
            pass



def twx(data, dataRange=[], cmap='gray', mode=None, titles=None):
    """
    Display and quickly switch between input images/video sequences
    Inputs:
        data - list of images/video sequences. Each image/video sequence can be a numpy array
            of up to 4 dimensions. For example:
            HxW or HxWx1 or HxWx3 size numpy array for image,
            HxWxN or Nx3xHxW size numpy array for video sequence
        dataRange - list of data ranges for each image/video sequence. 
            If list is shorter than the number of images/video sequences, 
            the last data range will be used for the remaining images/video sequences. 
            Each data range can be: list of either 0, 1 or 2 elements.
            - empty list [] - use [numpy.min(data), numpy.max(data)] intensity range
            - single element list [P] - use [numpy.percentile(data,P), numpy.percentile(data,100-P)] intensity range
            - two numbers [low, high] - use [low, high] intensity range
        cmap - colormap to use for displaying grayscale data, ignored for
                RGB data
        mode - string specifying the order of dimensions in the input data. It can be one of the following:
                'HW'   - Height x Width (grayscale image)
                'HWC'  - Height x Width x Channels (RGB image)
                'CHW'  - Channels x Height x Width (RGB image)
                'NHW'  - Number of frames x Height x Width (grayscale video)
                'NHWC' - Number of frames x Height x Width x Channels (RGB video)
                'NCHW' - Number of frames x Channels x Height x Width (RGB video)

                If mode is not provided, it will be auto-detected based on the shape of the first
                    image/video sequence in the input data. Auto-detection rules:
                    If the first image/video sequence has
                        - 2 dimensions: it is assumed to be 'HW'
                        - 3 dimensions, and the first or last dimension has size 3,
                            it is assumed to be 'CHW' or 'HWC' respectively, 
                            otherwise it is assumed to be 'NHW'
                        - 4 dimensions, and the second or the last dimension has size 3, 
                            it is assumed to be 'NCHW' or 'NHWC' respectively, 
                        - 4 dimensions, and the second or the last dimension has size 1,
                            it is assumed to be 'NHW'


    Key controls:
        1,2,3, ..., 9, 0 - switch between input images/videos
        up/down arrow keys - show next/previous frame
        a - display/hide colorbar
        m - change colormap
        c - show contrast adjustment window (not working yet)

    Usage examples:

    """

    if isinstance(data, dict):
        imageTitles = list(data.keys())
        images = list(data.values())
    else:
        if isinstance(data, (list, tuple)):
            images = list(data)
        elif isinstance(data, np.ndarray):
            images = [data, ]

        if titles is None:
            imageTitles = [str(x+1) for x in range(len(images))]
        else:
            imageTitles = titles

    nImages = len(images)
    
    if mode is None:
        # Try to autodetect from the first image
        shape = images[0].shape
        if images[0].ndim == 2:
            mode="HW"
        elif images[0].ndim == 3:
            if shape[0] == 3:
                mode="CHW"
            elif shape[2] == 3:
                mode="HWC"
            else:
                mode="NHW"
        elif images[0].ndim == 4:
            if shape[1] == 3:
                mode="NCHW"
            elif shape[3] == 3:
                mode="NHWC"
            elif shape[1] == 1:
                mode="NCHW"
            elif shape[3] == 1:
                mode="NHWC"
            else:
                raise ValueError(
                    f"Cannot auto-detect mode for 4D array with shape {shape}. "
                    "Expected second or last dimension to be 1 or 3. "
                    "Please specify mode explicitly."
                )

    def normalize_data(image, mode):
        if mode=='HW':
            image=image[None,:,:,None]
            frame_axis=0
            channel_axis=3
        elif mode=='CHW':
            image=image[None,:,:]
            frame_axis=0
            channel_axis=1
        elif mode=='HWC':
            image=image[None,:,:]
            frame_axis=0
            channel_axis=3
        elif mode == 'NHW':
            image=image[:,:,:,None]
            frame_axis=0
            channel_axis=3
        elif mode == 'HWN':
            image=image[:,:,:,None]
            frame_axis=2
            channel_axis=3   
        elif mode in ['NCHW', 'NHWC']:
            frame_axis=mode.find('N')
            channel_axis=mode.find('C')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        image = np.moveaxis(image,
                                (channel_axis, frame_axis), (-1, 0))
        
        return image
                
    images=[normalize_data(image, mode) for image in images]
    rgb = (images[0].shape[-1]==3)
    # print(images[i].shape)
    # ----------- parsing and seeting dataRange parameter  -----------------

    if not isinstance(dataRange, list):  # case when dataRange is a number
        dataRange = [[dataRange]]  # a -> [[a]]

    if len(dataRange) == 0:  # when dataRange is empty list []
        dataRange = [[0]]

    if not isinstance(dataRange[0], list):  # if it is a list of numbers
        dataRange = [dataRange]

    # at this moment dataRange should be list of lists
    for k,rng in enumerate(dataRange):
        if len(rng) == 0:
            rng = [0]
            dataRange[k] = rng

        percent = -1

        if len(rng) == 1:
            percent = rng[0]

        if percent >= 0:
            dataRange[k] = list(np.percentile(images[k][0, :, :, :],
                                           (percent, 100-percent)).astype(np.float32))
    
    if len(dataRange) != len(images):
        dataRange.extend([dataRange[-1]]*len(images))

    # ----------- END OF parsing and setting parameters  -----------------

    fig = plt.figure(FigureClass=TWXFigure)
    fig.setData(images, dataRange, cmap, imageTitles, rgb)
    plt.show(block=False)
    bring_to_front(fig)


class CyclicValues():
    def __init__(self, values):
        self._values = list(values)
        self._idx = 0

    def cycle(self):
        self._idx = (self._idx + 1) % len(self._values)

    def __call__(self):
        return self._values[self._idx]

    def value(self):
        return self._values[self._idx]

    def set_value(self, value):
        if value not in self._values:
            self._values.append(value)
        self._idx = self._values.index(value)


def HiddenProperty(prop_fn):
    hidden_attribute_name = "_{}".format(prop_fn.__name__)

    @property
    @functools.wraps(prop_fn)
    def check_attr(self):
        if not hasattr(self, hidden_attribute_name):
            setattr(self, hidden_attribute_name, prop_fn(self))
        return getattr(self, hidden_attribute_name).value()

    @check_attr.setter
    def check_attr(self, value):
        if not hasattr(self, hidden_attribute_name):
            setattr(self, hidden_attribute_name, prop_fn(self))
        getattr(self, hidden_attribute_name).set_value(value)

    return check_attr


#class Test():
#    def __init__(self):
#        self.a = 1
#        pass
#
#    @HiddenProperty
#    def gamma(self):
#        return CyclicValues([1, 2, 3, 4])

class TWXFigure(Figure):
    cmaps = ('gray', 'jet', 'plasma', 'hot')
    gammas = (1.0, 0.5, 0.25, 2.0, 4.0)

    isRGB = False

    def __init__(self, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)
        self.mAxesImage = None
        self.title = self.text(0.5, 0.95, '', ha='center')

    @HiddenProperty
    def gamma(self): return CyclicValues(TWXFigure.gammas)

    @HiddenProperty
    def cmap(self): return CyclicValues(TWXFigure.cmaps)

    def setData(self, data, dataRanges, cmap, imageTitles, isRGB):
        # self.fig = plt.figure()
        self.data = data
        self.dataRanges = dataRanges
        self.imageTitles = imageTitles
        self.isRGB = isRGB

        self.nImages = len(data)
        self.f_ax = 0
        self.h_ax = 1
        self.w_ax = 2
        self.c_ax = 3
        self.cmap = cmap

        # we need to set frame index before we set image index
        self.curFrameIdx = 0
        self.setCurrentImage(idx=0)
        self.connect()

    def connect(self):
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        #self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_key_press(self, event):
        if event.key in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            self.setCurrentImage((int(event.key)-1) % 10)

        elif event.key in {'up', 'down'}:  # {'right','left'}:
            self.setCurrentFrame(self.curFrameIdx +
                                 (1 if event.key == 'down' else -1))

        elif event.key == 'a':
            self.SwitchColorbar()

        elif event.key == 'ctrl+g':
            self._gamma.cycle()
            self.update_buffered_image()

        elif event.key == 'm':
            self.SwitchColormap()
        elif event.key == 'c':
            contrast_tool()

    def setCurrentImage(self, idx):
        if idx >= 0 and idx < self.nImages:
            self.curImIdx = idx
            self.curIm = self.data[self.curImIdx]
            self.nFrames = self.curIm.shape[self.f_ax]
            self.dataRange = self.dataRanges[self.curImIdx]
            self.setCurrentFrame(self.curFrameIdx)

    def setCurrentFrame(self, idx):
        self.curFrameIdx = idx % self.nFrames
        self.update_buffered_image()

    def update_buffered_image(self):
        """
        update_buffered_image should be used to update cached copy of
        current image. Cached copy is stored in self.curFrame.
        self.curFrame and is used any adjustments made to the current image,
        such  as brightness stretching or gamma correction.
        If only image display options have been updated without
        altering self.curFrame then call Draw() directly.
        """
        # if self.isRGB:
        #     im = self.curIm[self.curFrameIdx,...].squeeze()
        #     vmin, vmax = self.dataRange

        #     im = np.clip((im.astype(np.float32) - vmin)/(vmax-vmin), 0, 1)
        #     self.curFrame = im ** self.gamma
        # else:
        #     self.curFrame = self.curIm[self.curFrameIdx,...].squeeze()
        self.curFrame = self.curIm[self.curFrameIdx,...].squeeze()
        self.Draw()

    def Draw(self):

        vmin, vmax = self.dataRange
        self._norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)

        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)

        im = self.curFrame

        if self.isRGB:
            im=self._norm(im)

        if self.mAxesImage is None:
            self.mAxesImage = plt.imshow(im, cmap=self.cmap)
        else:
            self.mAxesImage.set_data(im)

        self.mAxesImage.set_norm(self._norm)

        logger.setLevel(old_level)

        self.title.set_text("Im:{:s} Frame: {:d}".format(
                                       self.imageTitles[self.curImIdx],
                                       self.curFrameIdx+1))
        self.canvas.draw()
        self.canvas.flush_events()

    def SwitchColormap(self):
        self._cmap.cycle()
        self.mAxesImage.set_cmap(self.cmap)
        self.canvas.draw()

    def SwitchColorbar(self):
        if self.mAxesImage.colorbar is None:
            # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            # plt.colorbar(cax=cax)
            # self.cbar = self.colorbar(self.mAxesImage, cax=cax)
            self.cbar = self.colorbar(self.mAxesImage)
        else:
            self.mAxesImage.colorbar.remove()
            self.subplots_adjust()
        self.canvas.draw()


#def roi_tool(current_ax):
#    def line_select_callback(eclick, erelease):
#        'eclick and erelease are the press and release events'
#        x1, y1 = eclick.xdata, eclick.ydata
#        x2, y2 = erelease.xdata, erelease.ydata
#        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
#        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
#
#
#    def toggle_selector(event):
#        print(' Key pressed.')
#        if event.key in ['W', 'w'] and toggle_selector.RS.active:
#            print(' RectangleSelector deactivated.')
#            toggle_selector.RS.set_active(False)
#        if event.key in ['B', 'b'] and not toggle_selector.RS.active:
#            print(' RectangleSelector activated.')
#            toggle_selector.RS.set_active(True)
#
#
#    toggle_selector.RS = RectangleSelector(
#                                    current_ax,
#                                    line_select_callback,
#                                    drawtype='box', useblit=True,
#                                    button=[1, 3],  # don't use middle button
#                                    minspanx=5, minspany=5,
#                                    spancoords='pixels',
#                                    interactive=True)
#    plt.connect('key_press_event', toggle_selector)
#    pass


def contrast_tool(targetfig=None):
    """
    Launch a subplot tool window for a figure.
    A :class:`matplotlib.widgets.SubplotTool` instance is returned.
    """
    tbar = rcParams['toolbar']  # turn off the navigation toolbar for the toolfig
    rcParams['toolbar'] = 'None'
    if targetfig is None:
        manager = plt.get_current_fig_manager()
        targetfig = manager.canvas.figure
    else:
        # find the manager for this figure
        for manager in _pylab_helpers.Gcf._activeQue:
            if manager.canvas.figure == targetfig:
                break
        else:
            raise RuntimeError('Could not find manager for targetfig')

    toolfig = plt.figure(figsize=(6, 3))
    toolfig.subplots_adjust(top=0.9)
    ret = ContrastTool(targetfig, toolfig)
    rcParams['toolbar'] = tbar
    _pylab_helpers.Gcf.set_active(manager)  # restore the current figure
    return ret


class ContrastTool(Widget):
    """
    A tool to adjust the subplot params of a :class:`matplotlib.figure.Figure`.
    """
    def __init__(self, targetfig, toolfig):
        """
        *targetfig*
            The figure instance to adjust.

        *toolfig*
            The figure instance to embed the subplot tool into. If
            *None*, a default figure will be created. If you are using
            this from the GUI
        """
        # FIXME: The docstring seems to just abruptly end without...

        self.targetfig = targetfig
        toolfig.subplots_adjust(left=0.2, right=0.9)

        class toolbarfmt:
            def __init__(self, slider):
                self.slider = slider

            def __call__(self, x, y):
                fmt = '%s=%s' % (self.slider.label.get_text(),
                                 self.slider.valfmt)
                return fmt % x

        self.axleft = toolfig.add_subplot(711)
        self.axleft.set_title('Click on slider to adjust subplot param')
        self.axleft.set_navigate(False)

        self.slidermin = Slider(self.axleft, 'min',
                                0, 1,
                                0,  # val
                                closedmax=False)
        self.slidermin.on_changed(self.funcmin)

        self.axbottom = toolfig.add_subplot(712)
        self.axbottom.set_navigate(False)
        self.slidermax = Slider(self.axbottom,
                                'max', 0, 1,
                                1,  # val
                                closedmax=False)
        self.slidermax.on_changed(self.funcmax)

        self.axright = toolfig.add_subplot(713)
        self.axright.set_navigate(False)
        self.sliderright = Slider(self.axright, 'bottom', 0, 1,
                                  targetfig.subplotpars.right,
                                  closedmin=False)
        self.sliderright.on_changed(self.funcbottom)

        self.axtop = toolfig.add_subplot(714)
        self.axtop.set_navigate(False)
        self.slidertop = Slider(self.axtop, 'top', 0, 1,
                                targetfig.subplotpars.top,
                                closedmin=False)
        self.slidertop.on_changed(self.functop)

        self.axwspace = toolfig.add_subplot(715)
        self.axwspace.set_navigate(False)
        self.sliderwspace = Slider(self.axwspace, 'wspace',
                                   0, 1, targetfig.subplotpars.wspace,
                                   closedmax=False)
        self.sliderwspace.on_changed(self.funcwspace)

        self.axhspace = toolfig.add_subplot(716)
        self.axhspace.set_navigate(False)
        self.sliderhspace = Slider(self.axhspace, 'hspace',
                                   0, 1, targetfig.subplotpars.hspace,
                                   closedmax=False)
        self.sliderhspace.on_changed(self.funchspace)

        # constraints
        self.slidermin.slidermax = self.slidermax
        self.slidermax.slidermin = self.slidermin

        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonreset = Button(bax, 'Reset')

        sliders = (self.slidermin, self.slidermax, self.sliderright,
                   self.slidertop, self.sliderwspace, self.sliderhspace,)

        def func(event):
            thisdrawon = self.drawon

            self.drawon = False

            # store the drawon state of each slider
            bs = []
            for slider in sliders:
                bs.append(slider.drawon)
                slider.drawon = False

            # reset the slider to the initial position
            for slider in sliders:
                slider.reset()

            # reset drawon
            for slider, b in zip(sliders, bs):
                slider.drawon = b

            # draw the canvas
            self.drawon = thisdrawon
            if self.drawon:
                toolfig.canvas.draw()
                self.targetfig.canvas.draw()

        # during reset there can be a temporary invalid state
        # depending on the order of the reset so we turn off
        # validation for the resetting
        validate = toolfig.subplotpars.validate
        toolfig.subplotpars.validate = False
        self.buttonreset.on_clicked(func)
        toolfig.subplotpars.validate = validate

    def setClim(self, vmin=None, vmax=None):
        self.targetfig.gca().get_images()[0].set_clim(vmin, vmax)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funcmin(self, val):
        self.setClim(vmin=val)

    def funcmax(self, val):
        self.setClim(vmax=val)

    def funcbottom(self, val):
        self.targetfig.gca().get_images()[0].set_clim(vmax=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def functop(self, val):
        self.targetfig.subplots_adjust(top=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val)
        if self.drawon:
            self.targetfig.canvas.draw()


# a = 0.8*np.ones((2, 3, 128, 256))
# a[:,:,:,128:]=0.2
# a = a*np.array([0.8,0.3, 0.1])[None,:, None,None]
# a[0]=a[0]*3

# twx(a[:,:,:,:], [0, 3])