import numpy as np
import pandas as pd
from sympy import Poly, Matrix
import pysindy as ps

from symm_ml.symbolic_utils import symb_jac_from_feat


def safe_coeff_monomial(poly, feat):
    '''
    Safe wrapper for sympy's poly.coeff_monomial, 
    returns 0 if feature does not exist in expression
    '''
    try:
        return poly.coeff_monomial(feat)
    except ValueError:
        return 0.0
    
def get_poly_coefs(poly, symb_feats):
    '''Extract a polynomial's coefficients from a set of symbolic features'''
    return np.array([safe_coeff_monomial(poly, feat) 
                     for feat in symb_feats], dtype = np.float64)
    
def symb_phi(phi_feats, coefs):
    '''Input features for a poly'''
    return Matrix(phi_feats).dot(Matrix(coefs))

def coef_lib_to_df(coefs_list, poly_lib):
    df = pd.DataFrame({'terms': poly_lib.get_feature_names()})
    
    for i, coefs in enumerate(coefs_list):
        df[f'coefs_{i}'] = coefs
    return df 

def get_lib_fn(input_dim, feature_dim, poly_deg, fn_cat):
    '''
    Get the coefficients and library for the functions:
        F = phi_d \circ y: R^m -> R
        Where 
            phi_d:  R^r -> R
                is a random polynomial of degree d on r variables,
            y: R^m -> R^r
                is a feature vector, either an orthogonal projection
                or quadratic shifts ||x - c_k||^2
    input_dim: (int)
        input dimension, m
    feature_dim: (int)
        feature dimension, r
    poly_deg: (int)
        polynomial degree for phi_d, d
    fn_cat: (str)
        function category, from ['ortho', 'quad']
            - 'ortho', y features are random orthogonal projections u_k^T x 
            - 'quad', y features are quadratic shifts ||x - c_k||^2
    
    Returns:
        coefs: array of coefficients for the poly_lib
        poly_lib: pysindy.PolynomialLibrary 
    '''
    
    # setup the library for phi_d
    phi_lib = ps.PolynomialLibrary(degree=poly_deg)
    phi_lib.fit(np.zeros((2,feature_dim)), np.zeros(2))
    phi_feats = phi_lib.get_feature_names()
    phi_symb_feats, _ = symb_jac_from_feat(phi_feats,feature_dim)

    # choose poly_lib degree w.r.t input variables based off features
    if fn_cat == 'ortho':
        poly_lib = ps.PolynomialLibrary(degree = poly_deg)
    elif fn_cat == 'quad':
        poly_lib = ps.PolynomialLibrary(degree = 2*poly_deg)
    else:
        raise KeyError(f'fn_cat {fn_cat} not in ["ortho", "quad"]')
    
    # setup the library for the final polynomial
    poly_lib.fit(np.zeros((2,input_dim)), np.zeros(2))
    poly_lib_feats = poly_lib.get_feature_names()
    poly_lib_symb_feats, _ = symb_jac_from_feat(poly_lib_feats, input_dim)

    x = poly_lib_symb_feats[1:input_dim+1] # symbolic input variables in R^m

    # randomly generate coefficients for phi_d 
    rand_phi_coefs = np.random.rand(len(phi_feats))

    phi = symb_phi(phi_symb_feats, rand_phi_coefs)

    # generate features
    if fn_cat == 'quad': 
        # random uniform on [-1,1]
        c = np.random.uniform(-1, 1, feature_dim*input_dim).reshape(feature_dim,input_dim)
        # c *= 0.5
        tmps = [Matrix(x - c[k]) for k in range(feature_dim)]
        y = [tmp.dot(tmp) for tmp in tmps]

    elif fn_cat == 'ortho':
        # random, unit-normally distributed vectors 
        rand_u = np.random.normal(size=(input_dim,feature_dim))
        rand_u = np.linalg.svd(rand_u)[0]
        y = [Matrix(x).dot(u) for u in rand_u[:feature_dim]]

    # define the final function by symbolic composition
    F=phi.subs({
        x[k]: y[k] for k in range(feature_dim)
    }, simultaneous=True)
    
    # extract the coefficients from F using the poly_lib features
    # (ensures proper ordering)
    coefs = get_poly_coefs(Poly(F), poly_lib_symb_feats)
    
    return coefs, poly_lib

