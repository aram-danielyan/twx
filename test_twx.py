import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch
import pytest

import twxtools._viewer as twx_module
from twxtools import twx
from twxtools._viewer import CyclicValues, TWXFigure


# --- Helper: run twx without bring_to_front (no AppKit in test) ---
@pytest.fixture(autouse=True)
def mock_bring_to_front():
    with patch.object(twx_module, 'bring_to_front'):
        yield


# ========================== Data input types ==========================

class TestDataInputTypes:
    def test_single_ndarray(self):
        img = np.random.rand(64, 64)
        twx(img)
        plt.close('all')

    def test_list_of_arrays(self):
        imgs = [np.random.rand(64, 64), np.random.rand(64, 64)]
        twx(imgs)
        plt.close('all')

    def test_tuple_of_arrays(self):
        imgs = (np.random.rand(64, 64), np.random.rand(64, 64))
        twx(imgs)
        plt.close('all')

    def test_dict_of_arrays(self):
        imgs = {'first': np.random.rand(64, 64), 'second': np.random.rand(64, 64)}
        twx(imgs)
        plt.close('all')

    def test_dict_uses_keys_as_titles(self):
        imgs = {'alpha': np.random.rand(32, 32), 'beta': np.random.rand(32, 32)}
        with patch.object(TWXFigure, 'setData', wraps=None) as mock_set:
            twx(imgs)
        mock_set.assert_called_once()
        titles_arg = mock_set.call_args[0][3]
        assert titles_arg == ['alpha', 'beta']
        plt.close('all')


# ========================== Titles parameter ==========================

class TestTitles:
    def test_default_titles_are_indices(self):
        imgs = [np.random.rand(32, 32)] * 3
        with patch.object(TWXFigure, 'setData', wraps=None) as mock_set:
            twx(imgs)
        titles_arg = mock_set.call_args[0][3]
        assert titles_arg == ['1', '2', '3']
        plt.close('all')

    def test_custom_titles(self):
        imgs = [np.random.rand(32, 32)] * 2
        with patch.object(TWXFigure, 'setData', wraps=None) as mock_set:
            twx(imgs, titles=['A', 'B'])
        titles_arg = mock_set.call_args[0][3]
        assert titles_arg == ['A', 'B']
        plt.close('all')

    def test_dict_ignores_titles_param(self):
        imgs = {'x': np.random.rand(32, 32)}
        with patch.object(TWXFigure, 'setData', wraps=None) as mock_set:
            twx(imgs, titles=['ignored'])
        titles_arg = mock_set.call_args[0][3]
        assert titles_arg == ['x']
        plt.close('all')


# ========================== Mode auto-detection ==========================

class TestModeAutoDetection:
    def test_2d_detected_as_HW(self):
        img = np.random.rand(64, 48)
        twx(img)
        fig = plt.gcf()
        # After normalize: (1, 64, 48, 1) -> shape[0]=frames=1
        assert fig.data[0].shape == (1, 64, 48, 1)
        plt.close('all')

    def test_3d_last_dim_3_detected_as_HWC(self):
        img = np.random.rand(64, 48, 3)
        twx(img)
        fig = plt.gcf()
        # HWC -> [None,:,:] -> (1,64,48,3) -> moveaxis(3,0)->(-1,0) -> (1,64,48,3)
        assert fig.data[0].shape[0] == 1  # 1 frame
        assert fig.data[0].shape[-1] == 3  # 3 channels
        plt.close('all')

    def test_3d_first_dim_3_detected_as_CHW(self):
        img = np.random.rand(3, 64, 48)
        twx(img)
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 1  # 1 frame
        assert fig.data[0].shape[-1] == 3  # 3 channels
        plt.close('all')

    def test_3d_no_dim_3_detected_as_NHW(self):
        vid = np.random.rand(10, 64, 48)
        twx(vid)
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 10  # 10 frames
        assert fig.data[0].shape[-1] == 1  # 1 channel (grayscale)
        plt.close('all')

    def test_4d_last_dim_3_detected_as_NHWC(self):
        vid = np.random.rand(5, 64, 48, 3)
        twx(vid)
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 5
        assert fig.data[0].shape[-1] == 3
        plt.close('all')

    def test_4d_second_dim_3_detected_as_NCHW(self):
        vid = np.random.rand(5, 3, 64, 48)
        twx(vid)
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 5
        assert fig.data[0].shape[-1] == 3
        plt.close('all')

    def test_4d_second_dim_1_detected_as_grayscale(self):
        vid = np.random.rand(5, 1, 64, 48)
        twx(vid)
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 5
        assert fig.data[0].shape[-1] == 1
        plt.close('all')

    def test_4d_last_dim_1_detected_as_grayscale(self):
        vid = np.random.rand(5, 64, 48, 1)
        twx(vid)
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 5
        assert fig.data[0].shape[-1] == 1
        plt.close('all')

    def test_4d_ambiguous_raises(self):
        vid = np.random.rand(5, 4, 64, 48)
        with pytest.raises(ValueError, match="Cannot auto-detect mode"):
            twx(vid)
        plt.close('all')


# ========================== Explicit mode ==========================

class TestExplicitMode:
    def test_mode_HW(self):
        img = np.random.rand(64, 48)
        twx(img, mode='HW')
        fig = plt.gcf()
        assert fig.data[0].shape == (1, 64, 48, 1)
        plt.close('all')

    def test_mode_NHW(self):
        vid = np.random.rand(10, 64, 48)
        twx(vid, mode='NHW')
        fig = plt.gcf()
        assert fig.data[0].shape == (10, 64, 48, 1)
        plt.close('all')

    def test_mode_NCHW(self):
        vid = np.random.rand(5, 3, 64, 48)
        twx(vid, mode='NCHW')
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 5
        assert fig.data[0].shape[-1] == 3
        plt.close('all')

    def test_mode_NHWC(self):
        vid = np.random.rand(5, 64, 48, 3)
        twx(vid, mode='NHWC')
        fig = plt.gcf()
        assert fig.data[0].shape[0] == 5
        assert fig.data[0].shape[-1] == 3
        plt.close('all')

    def test_unsupported_mode_raises(self):
        img = np.random.rand(64, 48)
        with pytest.raises(ValueError, match="Unsupported mode"):
            twx(img, mode='XYZ')
        plt.close('all')


# ========================== dataRange ==========================

class TestDataRange:
    def test_default_empty_range(self):
        img = np.random.rand(32, 32)
        twx(img)
        fig = plt.gcf()
        vmin, vmax = fig.dataRange
        assert vmin == pytest.approx(img.min(), abs=1e-5)
        assert vmax == pytest.approx(img.max(), abs=1e-5)
        plt.close('all')

    def test_explicit_range(self):
        img = np.random.rand(32, 32)
        twx(img, dataRange=[0.2, 0.8])
        fig = plt.gcf()
        assert fig.dataRange == [0.2, 0.8]
        plt.close('all')

    def test_percentile_range(self):
        img = np.random.rand(100, 100)
        twx(img, dataRange=[5])
        fig = plt.gcf()
        vmin, vmax = fig.dataRange
        expected_low = np.percentile(img, 5).astype(np.float32)
        expected_high = np.percentile(img, 95).astype(np.float32)
        assert vmin == pytest.approx(float(expected_low), abs=1e-5)
        assert vmax == pytest.approx(float(expected_high), abs=1e-5)
        plt.close('all')

    def test_scalar_datarange(self):
        img = np.random.rand(32, 32)
        twx(img, dataRange=5)
        fig = plt.gcf()
        # scalar 5 -> percentile(5, 95)
        assert len(fig.dataRange) == 2
        plt.close('all')

    def test_per_image_datarange(self):
        imgs = [np.random.rand(32, 32), np.random.rand(32, 32)]
        twx(imgs, dataRange=[[],[2]])
        fig = plt.gcf()
        # [] -> default percentile 0 (full min/max)
        assert fig.dataRanges[0] == [pytest.approx(imgs[0].min().astype(np.float32), abs=1e-5),
                                      pytest.approx(imgs[0].max().astype(np.float32), abs=1e-5)]
        # [2] -> percentile(2, 98)
        expected_low = np.percentile(imgs[1], 2).astype(np.float32)
        expected_high = np.percentile(imgs[1], 98).astype(np.float32)
        assert fig.dataRanges[1][0] == pytest.approx(float(expected_low), abs=1e-5)
        assert fig.dataRanges[1][1] == pytest.approx(float(expected_high), abs=1e-5)
        plt.close('all')

    def test_range_extended_for_multiple_images(self):
        imgs = [np.random.rand(32, 32), np.random.rand(32, 32)]
        twx(imgs, dataRange=[0.1, 0.9])
        fig = plt.gcf()
        # same range should be applied to both images
        assert fig.dataRanges[0] == [0.1, 0.9]
        assert fig.dataRanges[1] == [0.1, 0.9]
        plt.close('all')


# ========================== RGB detection ==========================

class TestRGBDetection:
    def test_grayscale_not_rgb(self):
        img = np.random.rand(64, 48)
        twx(img)
        fig = plt.gcf()
        assert fig.isRGB == False
        plt.close('all')

    def test_rgb_image(self):
        img = np.random.rand(64, 48, 3)
        twx(img)
        fig = plt.gcf()
        assert fig.isRGB == True
        plt.close('all')


# ========================== CyclicValues ==========================

class TestCyclicValues:
    def test_initial_value(self):
        cv = CyclicValues([10, 20, 30])
        assert cv.value() == 10
        assert cv() == 10

    def test_cycle(self):
        cv = CyclicValues([10, 20, 30])
        cv.cycle()
        assert cv.value() == 20
        cv.cycle()
        assert cv.value() == 30
        cv.cycle()
        assert cv.value() == 10  # wraps around

    def test_set_value_existing(self):
        cv = CyclicValues([10, 20, 30])
        cv.set_value(30)
        assert cv.value() == 30

    def test_set_value_new(self):
        cv = CyclicValues([10, 20])
        cv.set_value(99)
        assert cv.value() == 99
