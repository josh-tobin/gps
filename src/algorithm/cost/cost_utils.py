import numpy as np

def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.

    Args:
        wp: 
            A TxD matrix containing weights for each dimension
        d:
            NxTxD data to evaluate norm on
        Jd:
        Jdd:
        l1:
        l2:
        alpha:
    """
    #Matlab signature: function [l,lx,lxx] = evall1l2term(wp,d,Jd,Jdd,l1,l2,alpha)
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d*sqrtwp
    dscl = d*wp
    dscls = d*(wp**2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq**2, axis=1, keepdims=True)*l2 \
        + np.sqrt(alpha+np.sum(dscl**2, axis=1, keepdims=True))*l1
    # First order derivative terms.
    d1 = dscl*l2 + (dscls/np.sqrt(alpha+np.sum(dscl**2, axis=1, keepdims=True))*l1)
    lx = np.transpose(np.sum(Jd*np.expand_dims(d1, axis=2), axis=1, keepdims=True),[0,2,1])
    assert lx.shape[2] == 1
    lx = lx[:,:,0]

    #TODO: Second order terms.
    """
    psq = np.sqrt(alpha+np.sum(dscl**2,axis=1,keepdims=True));
    d2 = l1*(( np.eye(wp.shape[1])*(wp**2/psq)) - \
            (dscls/np.expand_dims(dscls, axis=2))*(psq**3)) \
         +l2*(wp*np.tile(eye(wp.shape[1]),[1,1,T]))
    sec = np.sum(np.transpose(d1,[2,0,1,3])*np.transpose(Jdd,[3,1,0,2]),axis=3, keepdims=True);
    #lxx = sum(sum(bsxfun(@times,bsxfun(@times,permute(Jd,[1 4 3 2]),permute(Jd,[4 1 3 5 2])),permute(d2,[4 5 3 1 2])),4),5) + ...
    #    0.5*sec + 0.5*permute(sec,[2 1 3]);
    """