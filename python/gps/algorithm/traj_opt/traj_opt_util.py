import numpy as np

def traj_distr_kl(new_mu, new_sigma, new_traj_distr, prev_traj_distr):
    """Compute KL divergence between new and the previos trajectory
    distributions.

    Args:
        new_mu: T x dX, mean of new trajectory distribution, computed from forward
        new_sigma: T x dX x dX, variance of new trajectory distribution, computed from forward
        new_traj_distr: A linear gaussian policy object, new distribution
        prev_traj_distr: A linear gaussian policy object, previous distribution
    Returns:
        kl_div: KL divergence between new and previous trajectories
    """
    # Constants
    T = new_mu.shape[0]
    dU = new_traj_distr.dU

    # Initialize vector of divergences for each time step
    kl_div = np.zeros(T)

    # Step through trajectory
    for t in range(T):
        # Fetch matrices and vectors from trajectory distributions
        mu_t = mu[t,:]
        K_prev = prev_traj_distr.K[t,:,:]
        K_new = new_traj_distr.K[t,:,:]
        k_prev = prev_traj_distr.k[t,:,:]
        k_new = new_traj_distr.k[t,:,:]
        chol_prev = prev_traj_distr.chol_pol_covar[t,:,:]
        chol_new = new_traj_distr.chol_pol_covar[t,:,:]

        # Compute log determinants and precision matrices
        logdet_prev = 2*sum(np.log(np.diag(chol_prev)))
        logdet_new = 2*sum(np.log(np.diag(chol_new)))
        prc_prev = np.lstsq(chol_prev,np.lstsq(chol_prev.T, np.eye(dU)))
        prc_new = np.lstsq(chol_new,np.lstsq(chol_new.T, np.eye(dU)))

        # Construct matrix, vector, and constants
        M_prev = np.c_[np.r_[K_prev.T.dot(prc_prev).dot(K_prev),-K_prev.T.dot(prc_prev)],
                       np.r_[-prc_prev.dot(K_prev), prc_prev]]
        M_new = np.c_[np.r_[K_new.T.dot(prc_new).dot(K_new),-K_new.T.dot(prc_new)],
                       np.r_[-prc_new.dot(K_new), prc_new]]
        v_prev = np.c_[K_prev.T.dot(prc_prev).dot(k_prev),
                       -prc_prev.dot(k_prev)]
        v_new = np.c_[K_new.T.dot(prc_new).dot(k_new),
                       -prc_new.dot(k_new)]
        c_prev = 0.5*k_prev.T.dot(prc_prev).dot(k_prev)
        c_new = 0.5*k_new.T.dot(prc_new).dot(k_new)

        # Compute KL divergence at timestep t
        kl_div[t] = (max(0, -0.5*mu_t.T.dot((M_new-M_prev)).dot(mu_t) -
            mu_t.T.dot((v_new-v_prev)) - 0.5*np.sum(sigma_t*(M_new-M_prev)) -
            0.5*logdet_new + 0.5*logdet_prev)) - c_new + c_prev

    # Add up divergences across time to get total divergence
    return np.sum(kl_div)

def bracketing_line_search(line_search_data, con, new_eta, min_eta):
    """Adjust eta using second order bracketed line search.

    Args:
        ls_data: Dictionary storing new and old constraint violations
                          and etas
        con: Constraint violotion amount
        new_eta: new values of eta
        min_eta: minimum value of eta
    Returns:
        ls_data: Dictionary of information (same as argument)
        eta: Found value of eta
    """

    # Adjust eta.
    # TODO - make constants
    if not ls_data or ((abs(ls_data['c1']-con) < 1e-8*abs(ls_data['c1']+con)) and
            (abs(ls_data['c2']-con) < 1e-8*abs(ls_data['c2']+con))):
        # Take initial step if we don't have multiple points already available.
        ls_data = {'c1':con, 'c2':con, 'e1':eta, 'e2':eta}
        if con < 0:  # Too little change.
            rate = abs(1.0/(eta*con))
            cng = min(max(rate*eta*con,-5),5)
            eta = exp(log(eta) + cng)
        else:  # Too much change.
            rate = 0.01
            eta = eta + con*rate
    else:
        # Choose two points to fit.
        leta = eta
        if (ls_data['c1'] <= con and ls_data['c2'] > con and
                ls_data['c1'] < 0 and ls_data['c2'] > 0):
            mid = 1
            if con < 0:
                c1 = con
                c2 = ls_data['c2']
                e1 = leta
                e2 = ls_data['e2']
            else:
                c1 = ls_data['c1']
                c2 = con
                e1 = ls_data['e1']
                e2 = leta
        else:
            mid = 0
            if (ls_data['c1'] < 0 and ls_data['c2'] < 0 and con < 0 and  # Too little change in all cases.
                    eta <= ls_data['e1'] and eta <= ls_data['e2'] and  # eta is decreasing.
                    ls_data['c1'] != ls_data['c2'] and
                    abs(ls_data['c2']-con)/(max(abs(ls_data['c2']),abs(con))*abs(np.log(ls_data['e2'])-np.log(eta))) < 1e-2):
                    # Change in con is small compared to change in eta.
                # If rate is changing very slowly, try jumping to lowest
                # possible eta.
                if not np.isnan(algorithm.params.fid_debug):
                    fprintf(algorithm.params.fid_debug,'Rate is changing slowly, jumping to minimum eta.\n')
                c1 = ls_data['c1']
                c2 = con
                e1 = ls_data['e1']
                e2 = leta
                ls_data['c1'] = c1
                ls_data['c2'] = c2
                ls_data['e1'] = e1
                ls_data['e2'] = e2
                eta = max(mineta,algorithm.params.min_eta)
                return
            else:
                if abs(ls_data['c1']) <= abs(ls_data['c2']):
                    if ls_data['c1'] < con:
                        c1 = ls_data['c1']
                        c2 = con
                        e1 = ls_data['e1']
                        e2 = leta
                    else:
                        c1 = con
                        c2 = ls_data['c1']
                        e1 = leta
                        e2 = ls_data['e1']
                else:
                    if ls_data['c2'] < con:
                        c1 = ls_data['c2']
                        c2 = con
                        e1 = ls_data['e2']
                        e2 = leta
                    else:
                        c1 = con
                        c2 = ls_data['c2']
                        e1 = leta
                        e2 = ls_data['e2']

        # First, try to perform a log-space fit.
        # Note that this may no longer be convex.
        lc1 = c1*e1
        lc2 = c2*e2
        le1 = np.log(e1)
        le2 = np.log(e2)

        # Fit quadratic.
        a = (lc2-lc1) / 2*(le2-le1)
        b = 0.5*(lc1+lc2-2*a*(le1+le2))

        # Decide whether we want to solve in the original space instead.
        # Use ratio of gradients.
        if (abs(np.log(abs(lc1/lc2))) > 2*abs(np.log(abs(c1/c2))) and
                c1*c2 < 0 and rand() < 0.5):
            solve_orig = 1 # Solve in the original if the ratio of gradients is very high in log space (with 50 percent probability).
        else:
            solve_orig = 0

        # Compute minimium.
        if a < 0 and not solve_orig:  # Concave, good to go.
            nlogeta = -b/(2*a)
        else:  # Solve in original space.
            if solve_orig and not np.isnan(algorithm.params.fid_debug):
                fprintf(algorithm.params.fid_debug,'Solving non-log problem due to ratio choice: %f %f\n',abs(np.log(abs(lc1/lc2))),abs(np.log(abs(c1/c2))))
            if not solve_orig and not np.isnan(algorithm.params.fid_debug):
                fprintf(algorithm.params.fid_debug,'Solving non-log problem due to a > 0: %f %f\n',abs(np.log(abs(lc1/lc2))),abs(np.log(abs(c1/c2))))

            # Fit quadratic.
            a = (c2 - c1)/(2*(e2 - e1))
            b = 0.5*(c1+c2-2*a*(e1+e2))
            if a < 0:  # Concave, good to go:
                nlogeta = np.log(max(-b/(2*a),1e-50))
            else:
                # Something is very wrong.
                if not np.isnan(algorithm.params.fid_debug):
                    fprintf(algorithm.params.fid_debug,'Dual is not concave!\n')
                rate = abs(1.0/(con*eta)) #0.001
                cng = min(max(rate*con*eta,-5),5)
                nlogeta = np.log(eta) + cng

        if not np.isnan(algorithm.params.fid_debug):
             fprintf(algorithm.params.fid_debug,'Bracket points: %f %f (%f %f) a=%f b=%f\n',e1,e2,c1,c2,a,b)

        # Bound change.
        if mid == 1:
            MAX_LOG_STEP = np.Inf
        else:
            MAX_LOG_STEP = 4

        if nlogeta > np.log(eta):
            nlogeta = np.log(eta) + min(nlogeta-np.log(eta),MAX_LOG_STEP)
        else:
            nlogeta = np.log(eta) + max(nlogeta-np.log(eta),-MAX_LOG_STEP)

        # Save old info and set new eta.
        ls_data['c1'] = c1
        ls_data['c2'] = c2
        ls_data['e1'] = e1
        ls_data['e2'] = e2
        # TODO - algorithm.params should change
        eta = max(exp(nlogeta),max(mineta,algorithm.params.min_eta))
        # TODO - return
