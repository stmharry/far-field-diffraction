import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from attr import attrs
from attr import attrib


@attrs
class Base(object):
    input = attrib(default=None)

    @property
    def wavelength(self):
        return self.input.wavelength

    def visualize(self):
        mag = np.square(np.abs(self.psi))
        mag = np.power(mag, 0.5)

        plt.imshow(mag.T)
        plt.savefig(f'{self.__class__.__name__}.png')


@attrs
class Grid(object):
    x = attrib(default=None)
    y = attrib(default=None)
    dim = attrib(default=None)
    size = attrib(default=None)

    def __attrs_post_init__(self):
        if (self.x is None) and (self.y is None):
            x = (
                np.arange(self.size, dtype=np.float32) -
                (self.size - 1) / 2) * self.dim
            self.x = x
            self.y = x

        elif (self.dim is None) and (self.size is None):
            self.dim = self.x[1] - self.x[0]
            self.size = len(self.x)

    @property
    def mesh(self):
        return np.meshgrid(self.x, self.y)


@attrs
class PlaneWaveFront(object):
    wavelength = attrib(default=1.24e-10)
    amplitude = attrib(default=1.0)
    grid_dim = attrib(default=1e-6)
    grid_size = attrib(default=1024)

    @property
    def grid(self):
        return Grid(dim=self.grid_dim, size=self.grid_size)

    @property
    def psi(self):
        return np.full_like(self.grid.mesh[0], self.amplitude)


@attrs
class Filter(Base):
    diameter = attrib(default=40e-6)

    @property
    def grid(self):
        return self.input.grid

    @property
    def filter(self):
        (X, Y) = self.input.grid.mesh

        X /= self.diameter
        Y /= self.diameter

        filter_ = np.logical_or.reduce([
            np.logical_and(
                np.abs(X - 3) < 0.5,
                np.abs(Y - 2) < 0.5),
            np.logical_and(
                np.abs(X + 1) < 0.5,
                np.abs(Y + 3) < 0.5),
            np.logical_and(
                np.abs(X - 1.5) < 0.5,
                np.abs(Y + 2.5) < 1.5)])
        return filter_.astype(np.float32)

    @property
    def psi(self):
        return self.input.psi * self.filter


@attrs
class ImageFilter(Base):
    image_path = attrib(default=None)

    @property
    def grid(self):
        return self.input.grid

    @property
    def filter(self):
        image_pil = PIL.Image.open(self.image_path)
        image_pil = image_pil.convert('L')
        image_pil = image_pil.resize((self.grid.size, self.grid.size))
        return np.asarray(image_pil)

    @property
    def psi(self):
        return self.input.psi * self.filter


@attrs
class FraunhofferPropagator(Base):
    subsample = attrib(default=2)

    @property
    def grid(self):
        grid = self.input.grid
        x = np.fft.fftfreq(grid.size, d=grid.dim)
        x = np.fft.fftshift(x) * self.wavelength

        return Grid(x=x, y=x)

    @property
    def psi(self):
        shape = np.asarray(self.input.psi.shape)
        pad_shape = shape * (self.subsample - 1)

        begin = (pad_shape[0] // 2, pad_shape[1] // 2)
        psi = np.pad(self.input.psi, [
            (begin[0], pad_shape[0] - begin[0]),
            (begin[1], pad_shape[1] - begin[1])])
        isp = np.fft.fft2(psi)
        isp = np.fft.fftshift(isp)
        isp = isp[(
            slice(begin[0], begin[0] + shape[0]),
            slice(begin[1], begin[1] + shape[1]))]

        return isp


def main():
    wf = PlaneWaveFront(
        wavelength=2e-10,
        grid_dim=1e-6,
        grid_size=256)

    # ft = Filter(
    #     input=wf,
    #     diameter=10e-6)

    ft = ImageFilter(
        input=wf,
        image_path='hex.jpg')

    pg = FraunhofferPropagator(
        input=ft,
        subsample=5)

    ft.visualize()
    pg.visualize()


if __name__ == '__main__':
    main()