import numpy as np
from . import Elements
from . import PoroElasticProperties as prop
import scipy as sp


def find_eltype(mesh):

    _, number_nodes = mesh.connectivity.shape

    if number_nodes == 3:
        return 'linear'
    elif number_nodes == 6:
        return 'quadratic'
    else:
        return 'undefined'


def assemble_stiffness_matrix(mesh, E_vec, nu_vec):
    # First we initiate the stiffnes matrix as an epty matrix
    # The stiffness matrix is of shape
    # 2 * mesh.number_nodes because of x and y displacement
    K = np.zeros((2 * mesh.number_nodes, 2 * mesh.number_nodes))

    # We then obtain the element type
    eltype = find_eltype(mesh)

    # In agreement with the possibility of having multiple materials
    # we ensure to have the correct vector of elastic parameters.
    if np.isscalar(E_vec):
        E_vec = [E_vec]
    if np.isscalar(nu_vec):
        nu_vec = [nu_vec]

    # We finally loop over all elements
    for el_id in range(mesh.number_els):
        # Getting the material type and the corresponding elastic parameters
        mat_id = mesh.id[el_id]  # TODO
        E = E_vec[mat_id]
        nu = nu_vec[mat_id]

        # Note that for ease of coding the stiffness matrix is usually coded
        # using the bulk and shear modulus rather than young's modulus and
        # poisson's ratio. We transform here from one to the other.
        k = prop.bulk_modulus(E, nu)
        g = prop.shear_modulus(E, nu)

        # We get the elastic isotropic stiffness matrix for the element
        D = elastic_isotropic_stiffness(k, g, simultype=mesh.simultype)

        # We now access the nodes of the element and the corresponding
        # degrees of freedom (DOF)
        # Complete below
        node_indices = mesh.connectivity[el_id]  # TODO
        n_dof = np.vstack(
            [2 * node_indices, 2 * node_indices + 1]
        ).reshape(-1, order='F')

        # we get the coordinates of the nodes
        X = mesh.nodes[node_indices]  # TODO

        # We can now obtain the elements isoparametric representation
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # And can finally get the elements stiffness matrix.
        # This matrix depends on the elements elastic matrix
        K_el = elt.element_stiffness_matrix(D)

        # Finally, we put the components back into the global system at the
        # correctposition
        for i, ni in enumerate(n_dof):
            for j, nj in enumerate(n_dof):
                K[ni, nj] += K_el[i, j]  # TODO

    return sp.sparse.csc_matrix(K)


def project_flux(mesh, cond_vec, nodal_field, M=None, return_M=False):
    """
    Function to project the flux (derivatives of the nodal field) at the
    nodes of the mesh from the knowledge
    of the solution at the nodes q== - Cond Grad nodal_field

    Parameters
    ----------
    cond_vec : float | np.ndarray
        scalar or vector for conductivity
    nodal_field : np.ndarray
        vector containing the field at the nodes of the whole mesh

    Returns
    -------
    np.ndarray
        an array 'q' of shape (dim, number of nodes)
    """

    if len(nodal_field.shape) == 1:
        nodal_field = nodal_field[:, None]

    if np.isscalar(cond_vec):
        cond_vec = [cond_vec]

    # Step 1 : creation of the mesh matrix
    if M is None:
        rho_vec = np.full(len(cond_vec), 1.0)
        M = assemble_mass_matrix(mesh, rho_vec)

    eltype = find_eltype(mesh)

    # Step 2 : creating an empty array containing global nodal forces
    if mesh.simultype == '2D':
        nodal_force = np.zeros([mesh.dim, mesh.number_nodes])
    else:
        raise ValueError('Not implemented yet')

    # Step 3 : iterating through the mesh
    for el_id in range(mesh.number_els):
        # based on the material ID, we access the conductivity of the material
        mat_id = mesh.id[el_id]
        cond = cond_vec[mat_id]

        # Step 3.1 : access the node indices of the element
        node_indices = mesh.connectivity[el_id]  # TODO

        # Step 3.2 : get the coordinates of the nodes and construct an element
        X = mesh.nodes[node_indices]  # TODO
        element = Elements.Triangle(X, eltype, mesh.simultype)

        # Step 3.3 : assess nodal values corresponding to the element
        elt_heads = nodal_field[node_indices]  # TODO

        # Step 3.4 : Nodal forces caused by the fluxes at integration points
        nodal_force_per_elem = element.project_element_flux(cond, elt_heads)  # TODO

        # Step 3.5 : elemental forces => global nodal forces array
        nodal_force[:, node_indices] += nodal_force_per_elem  # TODO

    # Step 4 : initiate an empty array for the nodal fluxes
    nodal_flux = np.zeros_like(nodal_force)

    # Step 5 : obtain nodal fluxes in each direction by solving the system
    # mass_matrix * nodal_flux_vector = nodal_force_vector
    # Step 5.1 : get the inverse of the mass matrix
    Minv = sp.sparse.linalg.splu(M)
    for i in range(mesh.dim):
        # step 5.2 : Fast solve using the scipy method
        nodal_flux[i] = Minv.solve(nodal_force[i].T)

    if return_M:
        return nodal_flux, M
    else:
        return nodal_flux


def project_stress(mesh, E_vec, nu_vec, displacement, M=None, return_M=False):
    """
    Function to project the stress (derivatives of the nodal field) at the
    nodes of the mesh from the knowledge
    of the solution at the nodes

    Parameters
    ----------
    mesh : Mesh
        a mesh object giving the mesh of the problem
    E_vec : float | np.ndarray
        scalar or vector containing the Young's modulus
    nu_vec : float | np.ndarray
        scalar or vector containing Poisson's ratio
    displacement : np.ndarray
        vector containing the field at the nodes displacement

    Returns
    -------
    np.ndarray
        an array 'f' of shape (n_stresses, number of nodes)
        containing the stresses at the element level
        where n_stresses is the number of stresses
        you have (3 for 2D, 4 for axisymmetric)
    """

    # We get the element type
    eltype = find_eltype(mesh)

    # If the provided material parameters are scalars
    # we transform them in the correct sized array
    if ...:  # TODO
        ...

    if ...:  # TODO
        ...

    if np.isscalar(E_vec):
        E_vec = [E_vec]
    if np.isscalar(nu_vec):
        nu_vec = [nu_vec]

    # Step 1 : creation of the mesh matrix
    # In case M (the Mass matrix) was already computed,
    # to see for example stress evolution over time
    if M is None:
        M = assemble_mass_matrix(mesh, 1.0)

    # Step 2 : creating an empty array containing global nodal forces
    if mesh.simultype == '2D':
        nodal_force = np.zeros((3, mesh.number_nodes))
    elif mesh.simultype == 'axis':
        nodal_force = np.zeros((4, mesh.number_nodes))
    else:
        raise ValueError('Type not implemented yet')

    # Step 3 : iterating through the mesh
    for el_id in range(mesh.number_els):
        # based on the material ID, we access the material properties
        # Complete below
        # mat_id =
        # E =
        #  nu =
        mat_id = mesh.id[el_id]
        E = E_vec[mat_id]
        nu = nu_vec[mat_id]

        # we need to transform the properties to the bulk (k) and shear
        # modulus (g)
        # Complete below
        # k =
        # g =
        k = prop.bulk_modulus(E, nu)
        g = prop.shear_modulus(E, nu)

        # We want to obtain the elastic stiffness matrix
        D = elastic_isotropic_stiffness(k, g, mesh.simultype)

        # Step 3.1 : access the node indices of the
        # element and degrees of freedom
        # Complete below
        # n_e =
        # n_dof =
        n_e = mesh.connectivity[el_id]
        n_dof = np.vstack([2 * n_e, 2 * n_e + 1]).reshape(-1, order='F')

        # Complete below
        # Step 3.2 : get the coordinates of the nodes and construct an element
        X = mesh.nodes[n_e]

        # Step 3.3 : assemble the element
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # Step 3.4 : assess nodal values corresponding to the element
        elt_displacement = displacement[n_dof]

        # Step 3.5 : Nodal forces caused by the fluxes at integration points
        nodal_force_per_elem = elt.project_element_stress(D, elt_displacement)

        # Step 3.6 : elemental forces => global nodal forces array
        nodal_force[:, n_e] += nodal_force_per_elem.T

    # Step 4 : initiate an empty array for the nodal stresses
    f_out = np.zeros_like(nodal_force)

    # Step 5 : obtain nodal fluxes in each direction by solving the
    # system mass_matrix * nodal_flux_vector = nodal_force_vector
    # Step 5.1 : get the inverse of the mass matrix
    Minv = sp.sparse.linalg.splu(M)
    for i in range(mesh.dim):
        # step 5.2 : Fast solve using the scipy method
        f_out[i] = Minv.solve(nodal_force[i].T)

    # Step 6 : return all necessary information
    if return_M:
        return f_out, M
    else:
        return f_out


def elastic_isotropic_stiffness(k, g, simultype='2D'):
    La = k + (4. / 3.) * g
    Lb = k - (2. / 3.) * g

    if simultype == '2D':
        D = np.array([[La, Lb, 0],
                      [Lb, La, 0],
                      [0, 0, g]])

    elif simultype == 'axis':
        D = np.array([[La, Lb, 0, Lb],
                      [Lb, La, 0, Lb],
                      [0, 0, g, 0],
                      [Lb, Lb, 0, La]])

    else:
        raise ValueError('Simulation type not implemented yet')

    return D


def assemble_mass_matrix(mesh, rho_vec):
    """
    Function to assemble the mass matri of the system

    Parameters
    ----------
    mesh : Mesh
        a mesh object giving the mesh of the problem
    rho_vec : float | np.ndarray
        a scalar or vector containing the density of the materials

    Returns
    -------
    np.ndarray
        the mass matrix of the system of size (number nodes, number nodes)
    """

    # We get the element type
    eltype = find_eltype(mesh)

    # If the provided material parameter are scalars
    # we transform them in the correct sized array
    if np.isscalar(rho_vec):
        rho_vec = [rho_vec]

    # creation of the empty matrix
    M = np.zeros([mesh.number_nodes, mesh.number_nodes])

    # we now loop over all the elements to obtain a the mass matrix
    for el_id in range(mesh.number_els):

        # we access the density of the material by its ID
        mat_id = mesh.id[el_id]
        rho_e = rho_vec[mat_id]

        # access the node indices of the element
        node_indices = mesh.connectivity[el_id]
        # complete the code below
        # node_indices =

        # get the coordinates of the nodes and construct an element
        X = mesh.nodes[node_indices]
        # complete the code below
        # X = mesh.
        element = Elements.Triangle(X, eltype, mesh.simultype)

        # construct an elemental mass matrix
        M_el = element.element_mass_matrix(rho_e)
        # complete the code below
        # M_el = element.

        # aggregate elemental matrices into the global mass matrix
        for i, ni in enumerate(node_indices):
            for j, nj in enumerate(node_indices):
                M[ni, nj] += M_el[i, j]
                # complete the code below
                # M[ , ] += M_el[ , ]

    return sp.sparse.csc_matrix(M)


def assemble_conductivity_matrix(mesh, cond):
    """
    Function to assemble the conductivity matrix of the system

    Parameters
    ----------
    mesh : Mesh
        one of our mesh objects
    cond : float | np.ndarray
        the scalar (for uniform permeabilit) or array containing
        the conductivity of the material(s)

    outputs:
      - C :: the assembled conductivity matrix for the complete system
    """

    # we pre-define an empty matrix
    C = np.zeros((mesh.number_nodes, mesh.number_nodes))
    eltype = find_eltype(mesh)

    # we want to ensure the conductivity to be accessible by index
    if np.isscalar(cond):
        cond = [cond]

    # we loop over all the elements
    for el_id in range(mesh.number_els):

        # we access the conductivity of the element by its ID
        mat_id = mesh.id[el_id]
        cond_e = cond[mat_id]

        # access the node indices of the element
        node_indices = mesh.connectivity[el_id]
        # complete the code below
        # node_indices =

        # get the coordinates of the nodes and construct an element
        X = mesh.nodes[node_indices]
        # complete the code below
        # X = mesh.

        # we define the element
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # construct the element conductivity matrix
        C_el = elt.element_conductivity_matrix(cond_e)
        # complete the code below
        # C_el = element.

        # We assemble the element wise component into
        # the global conductivity matrix
        for i, ni in enumerate(node_indices):
            for j, nj in enumerate(node_indices):
                C[ni, nj] += C_el[i, j]  # TODO

    return sp.sparse.csc_matrix(C)


def assemble_coupling_matrix(mesh, alpha_vec):
    """
    Function to assemble the coupling matrix of the system

    Parameters
    ----------
    mesh : Mesh
        one of our mesh objects
    alpha_Vec : float | np.ndarray
        the scalar (for uniform permeabilit) or array containing
        the transmissivity of the material(s)

    outputs:
      - C :: the assembled coupling matrix for the complete system
    """

    # we pre-define an empty matrix
    C = np.zeros((2 * mesh.number_nodes, mesh.number_nodes))

    # We get the element type
    eltype = find_eltype(mesh)

    # If the provided material parameter are scalars
    # we transform them in the correct sized array
    if np.isscalar(alpha_vec):
        alpha_vec = [alpha_vec]

    # we now loop over all the elements to obtain a the mass matrix
    for e in range(mesh.number_els):

        # we access the transmissivity of the material by its ID
        mat_id = mesh.id[e]
        alpha = alpha_vec[mat_id]

        # access the nodes index of element e
        n_e = mesh.connectivity[e]
        n_dof = np.vstack([2 * n_e, 2 * n_e + 1]).reshape(-1, order='F')

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]  # TODO

        # create an element with its coordinates, type and simulation type
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # build an elementary coupling matrix
        ce_el = elt.element_coupling_matrix(alpha)  # TODO

        # fill in the global coupling matrix
        for i, ni in enumerate(n_dof):
            for j, nj in enumerate(n_e):
                C[ni, nj] += ce_el[i, j]  # TODO
    return sp.sparse.csc_matrix(C)


def set_stress_field(mesh, stress_field, applied_nodes=None):

    S = np.zeros(2 * mesh.number_nodes)
    eltype = find_eltype(mesh)

    # we want to find nodes with applied stress
    if applied_nodes is None:
        # if not specified, we apply the stress to the entire domain
        applied_nodes = np.arange(mesh.number_nodes)

    # we find the elements where every node has applied stress
    il = np.isin(mesh.connectivity, applied_nodes)
    elt_line = np.argwhere(il.sum(axis=1) == mesh.connectivity.shape[1])[:, 0]

    for e in elt_line:

        n_e = mesh.connectivity[e]
        n_dof = np.vstack([2 * n_e, 2 * n_e + 1]).reshape(-1, order='F')
        X = mesh.nodes[n_e]

        elt = Elements.Triangle(X, eltype, mesh.simultype)
        S_el = elt.element_stress_field(stress_field)
        S[n_dof] += S_el

    return S


def assemble_tractions_over_line(mesh, node_list, traction):

    # we obtain the element type from the mesh
    eltype = find_eltype(mesh)

    # getting the indexes of the nodes we need to address
    il = np.isin(mesh.connectivity, node_list)

    # we want to find the number of nodes lying on one side of every triangle
    if eltype == 'linear':
        n = 2
    elif eltype == 'quadratic':
        n = 3
    else:
        raise ValueError('Not implemented yet')

    # We identify all elements which have one side on the line
    elt_line = np.argwhere(il.sum(axis=1) == n)[:, 0]

    # we prepare the output force vectore as a list of two entries per node
    # corresponding to the x and y components of the vector
    f = np.zeros(2 * mesh.number_nodes)

    # We now loop over the elements with an edge on the boundary
    for i, e in enumerate(elt_line):
        # Number of the node on the line
        nn_l = il[e]

        # We get the equivalent global indices of the node
        global_nodes = mesh.connectivity[e, nn_l]
        global_dof = np.array([global_nodes * 2, global_nodes * 2 + 1]).T

        # and we get the coordinates of the node
        X = mesh.nodes[global_nodes]

        # Now we generate the line segment between the two nodes
        seg_xi, seg_yi = np.argsort(X, axis=0).T
        segx, segy = X[seg_xi, 0], X[seg_yi, 1]

        # The element consists of two elements,
        # one is the projection on the x-axis
        # the second is the projection on the y-axis
        elt_x = Elements.Segment(segx, eltype=eltype, simultype=mesh.simultype)
        elt_y = Elements.Segment(segy, eltype=eltype, simultype=mesh.simultype)

        # Now we calculate the corresponding forces using
        # the perpendicular traction to the element.
        fs = elt_y.neumann(traction[0])
        fn = elt_x.neumann(traction[1])

        # We can asselmble our global force vector
        f[global_dof[seg_yi, 0]] += fs
        f[global_dof[seg_xi, 1]] += fn

    # And finally return it.
    return f
