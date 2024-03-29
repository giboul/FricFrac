import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class Mesh:

    def __init__(self, meshio_mesh, cell_type='triangle', simultype='2D'):

        self.cell_type = cell_type

        self._parse_mesh(meshio_mesh)
        self.simultype = simultype

    def _parse_mesh(self, out):

        # we initially take the connectivity from the meshio object
        connectivity = out.cells_dict[self.cell_type].astype(int)

        # we need to filter the unused nodes that are residuals from pygmsh
        # did not find native way to do it
        used_nodes = np.zeros(len(out.points), dtype=bool)
        used_nodes[np.unique(connectivity)] = True
        self.dim = 2
        nodes = out.points[used_nodes, :self.dim]

        # we create an index map to reflect the new node indices
        index_map = np.zeros(len(out.points), dtype=int)
        index_map[used_nodes] = np.arange(len(nodes))
        new_connectivity = index_map[connectivity]

        # we now parse the arguments to the mesh object
        self.nodes = nodes
        self.connectivity = new_connectivity
        # number of elements
        self.number_els = len(self.connectivity)
        # number of nodes
        self.number_nodes = len(self.nodes)

        # material id
        self.id = np.zeros(self.number_els).astype(int)

    def plot(self, colorvalues=None, meshcolor='k', ax=None, box=True,
             triplot_kw=None, zoombox_kw=None, **tripcolor_kw):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if triplot_kw is None:
            triplot_kw = dict()
        triplot_kw = dict(ls='-.', lw=0.5) | triplot_kw

        ax.triplot(
            *self.nodes.T,
            self.connectivity[:, :3],
            c=meshcolor,
            **triplot_kw
        )

        if box is False:
            plt.box(False)
            plt.xticks([])
            plt.yticks([])

        if colorvalues is not None:
            im = ax.tripcolor(
                *self.nodes.T,
                self.connectivity[:, :3],
                colorvalues,
                **(tripcolor_kw or dict())
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.2, pad=0.1)
            cb = plt.colorbar(im, cax=cax)
        else:
            cb = None

        if zoombox_kw is not None:
            axins = self.zoom_plot(ax=ax, **zoombox_kw)

        ax.set_aspect("equal")
        return fig, ax, cb

    def zoom_plot(self,
                   zoom_xys=((None, None), (None, None)),
                   box_xys=((0.3, 0.8), (0.3, 0.8)),
                   c='k',
                   lw=0.5,
                   ls='-.',
                   alpha=1,
                   ax=None):

        if ax is None:
            ax = plt.gca()

        (x0, x1), (y0, y1) = box_xys
        xlim, ylim = zoom_xys

        axins = ax.inset_axes(
            [x0, y0, x1-x0, y1-y0],
            xlim=xlim, ylim=ylim,
            xticks=[], yticks=[]
        )
        _, connectors = ax.indicate_inset_zoom(axins, edgecolor="k", alpha=alpha)
        axins.triplot(*self.nodes.T, self.connectivity[:, :3], c=c, lw=lw, ls=ls)
        for c in connectors:
            c.set(alpha=1)

        axins.set_aspect("equal")
        
        return axins