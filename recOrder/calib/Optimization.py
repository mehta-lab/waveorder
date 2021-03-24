import numpy as np
from scipy import optimize
import os, sys
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)

from recOrder.recOrder.calib.CoreFunctions import snap_image, set_lc, get_lc, set_lc_state

def optimize_brent(calib, lca_bound_, lcb_bound_, reference, thresh, n_iter=3, mode=None, simul=False):
    """
    call scipy.optimize on the opt_lc function with arguments

    each iteration loop optimizes on LCA and LCB once, in sequence

    the loop will run until the percent error between the
    intensity at LCA/LCB value and the reference is below the threshold

    Bounds are around CURRENT LCA/LCB values.

    :param lca_bound_: float
        half the range to restrict the search for LCA
    :param lcb_bound_: float
        half the range to restrict the search for LCB
    :param reference: float
        optimize difference between opt_lc output and this value
        typically the current intensity value or best intensity value
    :param thresh: int
        The error percentage threshold to reach before
        stopping optimization

        %Error = fval/Reference * 100

    :return:
    """

    converged = False
    iteration = 1
    calib.inten = []
    optimal = []
    # Run until threshold is met
    while not converged:
        if calib.print_details:
            print(f'iteration: {iteration}')

        current_lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
        current_lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])

        # check that bounds don't exceed range of LC
        lca_lower_bound = 0.01 if (current_lca - lca_bound_) <= 0.01 else current_lca - lca_bound_
        lca_upper_bound = 1.6 if (current_lca + lca_bound_) >= 1.6 else current_lca + lca_bound_

        lcb_lower_bound = 0.01 if current_lcb - lcb_bound_ <= 0.01 else current_lcb - lcb_bound_
        lcb_upper_bound = 1.6 if current_lcb + lcb_bound_ >= 1.6 else current_lcb + lcb_bound_

        """
        xopt = optimal value of LC
        fval = output of opt_lc (mean intensity)
        ierr = error flag
        numfunc = number of function calls
        """

        if mode == '60' or mode == '120':

            # optimize lca
            xopt0, fval1, ierr0, numfunc0 = optimize.fminbound(calib.opt_lc_cons,
                                                               x1=lca_lower_bound,
                                                               x2=lca_upper_bound,
                                                               disp=0,
                                                               args=(reference,
                                                                     mode), full_output=True)

            set_lc(calib.mmc, xopt0, calib.PROPERTIES['LCA'])
            swing = (calib.lca_ext - xopt0) * calib.ratio
        
            if mode == '60':
                lcb = calib.lcb_ext + swing
                set_lc(calib.mmc, lcb, calib.PROPERTIES['LCB'])

            if mode == '120':
                lcb = calib.lcb_ext - swing
                set_lc(calib.mmc, lcb, calib.PROPERTIES['LCB'])
            
            optimal.append([fval1, xopt0, lcb])
            # Calculate Percent Error for LCA
            difference = (fval1 / reference) * 100
            
            if calib.print_details:
                print('\tOptimizing lca w/ constrained lcb ...')
                print(f"\tlca = {xopt0:.5f}")
                print(f'\tlcb = {lcb:.5f}')
                print(f'\tIntensity = {fval1 + reference}')
                print(f'\tIntensity Difference = {difference:.7f}%')

        elif mode == '90':

            xopt0, fval0, ierr0, numfunc0 = optimize.fminbound(calib.opt_lc,
                                                               x1=lca_lower_bound,
                                                               x2=lca_upper_bound,
                                                               disp=0,
                                                               args=(
                                                                   calib.PROPERTIES['LCA'],
                                                                   reference),
                                                               full_output=True)

            # Calculate Percent Error for LCA
            difference = (fval0 / reference) * 100
            lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])
            if calib.print_details:
                print('\tOptimize lca ...')
                print(f"\tlca = {xopt0:.5f}")
                print(f"\tlcb = {lcb:.5f}")
                print(f'\tIntensity = {fval0 + reference}')
                print(f'\tIntensity Difference = {difference:.7f}%')

            optimal.append([fval0, xopt0, lcb])

        elif mode == '45' or mode == '135':
            # optimize lcb
            xopt1, fval1, ierr1, numfunc1 = optimize.fminbound(calib.opt_lc,
                                                               x1=lcb_lower_bound,
                                                               x2=lcb_upper_bound,
                                                               disp=0,
                                                               args=(
                                                                   calib.PROPERTIES['LCB'],
                                                                   reference),
                                                               full_output=True)
            lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
            optimal.append([fval1, lca, xopt1])
            # Calculate Percent Error for LCB
            difference = fval1 / reference * 100
            
            if calib.print_details:
                print('\tOptimize lcb ...')
                print(f"\tlca = {lca:.5f}")
                print(f"\tlcb = {xopt1:.5f}")
                print(f'\tIntensity = {fval1 + reference}')
                print(f'\tIntensity Difference = {difference:.7f}%')
        
        else:
            
            xopt0, fval0, ierr0, numfunc0 = optimize.fminbound(calib.opt_lc,
                                                               x1=lca_lower_bound,
                                                               x2=lca_upper_bound,
                                                               disp=0,
                                                               args=(
                                                                   calib.PROPERTIES['LCA'],
                                                                   reference),
                                                               full_output=True)

            # Calculate Percent Error for LCA
            
            set_lc(calib.mmc, xopt0, calib.PROPERTIES['LCA'])
            difference = (fval0 / reference) * 100
            lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])
            if calib.print_details:
                print('\tOptimize lca ...')
                print(f"\tlca = {xopt0:.5f}")
                print(f'\tlcb = {lcb:.5f}')
                print(f'\tIntensity = {fval0 + reference}')
                print(f'\tIntensity Difference = {difference:.7f}%\n')


            optimal.append([fval0, xopt0, lcb])
            
            xopt1, fval1, ierr1, numfunc1 = optimize.fminbound(calib.opt_lc,
                                                               x1=lcb_lower_bound,
                                                               x2=lcb_upper_bound,
                                                               disp=0,
                                                               args=(
                                                                   calib.PROPERTIES['LCB'],
                                                                   reference),
                                                               full_output=True)
            
            set_lc(calib.mmc, xopt1, calib.PROPERTIES['LCB'])
            lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
            optimal.append([fval1, lca, xopt1])
            # Calculate Percent Error for LCB
            difference = fval1 / reference * 100
            if calib.print_details:
                print('\tOptimize lcb ...')
                print(f'\tlca = {lca:.5f}')
                print(f"\tlcb = {xopt1:.5f}")
                print(f'\tIntensity = {fval1 + reference}')
                print(f'\tIntensity Difference = {difference:.7f}%')
            

        # if both LCA and LCB meet threshold, stop
        if difference <= thresh:
            converged = True
            optimal = np.asarray(optimal)
            set_lc(calib.mmc, float(optimal[-1, 1]), calib.PROPERTIES['LCA'])
            set_lc(calib.mmc, float(optimal[-1, 2]), calib.PROPERTIES['LCB'])
            
            fval = optimal[-1,0]

            return float(fval)+reference
            

        # if loop preforms more than n_iter iterations, stop
        elif iteration >= n_iter:
            if calib.print_details:
                print(f'Exceeded {n_iter} Iterations: Search discontinuing')
            converged = True
            optimal = np.asarray(optimal)
            opt = np.where(optimal == np.min(np.abs(optimal[:, 0])))[0]

            if calib.print_details:
                print(f'Lowest Inten: {optimal[opt, 0]}, lca = {optimal[opt, 1]}, lcb = {optimal[opt, 2]}')
                
            set_lc(calib.mmc, float(optimal[opt, 1]), calib.PROPERTIES['LCA'])
            set_lc(calib.mmc, float(optimal[opt, 2]), calib.PROPERTIES['LCB'])
            fval = optimal[opt, 0]

            return float(fval)+reference

        iteration += 1




def optimize_grid(calib, a_min, a_max, b_min, b_max, step):
    """
    Exhaustive Search method

    Finds the minimum intensity value for a given
    grid of LCA,LCB values

    :param a_min: float
        Minimum value of LCA
    :param a_max: float
        Maximum value of LCA
    :param b_min: float
        Minimum value of LCB
    :param b_max: float
        Maximum value of LCB
    :param step: float
        step size of the grid between max/min values


    :return best_lca: float
        LCA value corresponding to lowest mean Intensity
    :return best_lcb: float
        LCB value corresponding to lowest mean Intensity
    :return min_int: float
        Lowest value of mean Intensity
    """

    min_int = 65536
    better_lca = -1
    better_lcb = -1
    
#     current_lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
#     current_lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])

#     # check that bounds don't exceed range of LC
#     a_min = 0.01 if (current_lca - a_min) <= 0.01 else current_lca - a_min
#     a_max = 1.6 if (current_lca + a_max) >= 1.6 else current_lca + a_max

#     b_min = 0.01 if current_lcb - b_min <= 0.01 else current_lcb - b_min
#     b_max = 1.6 if current_lcb + b_max >= 1.6 else current_lcb + b_max

    # coarse search
    for lca in np.arange(a_min, a_max, step):
        for lcb in np.arange(b_min, b_max, step):
            
            set_lc(calib.mmc, lca, calib.PROPERTIES['LCA'])
            set_lc(calib.mmc, lcb, calib.PROPERTIES['LCB'])

            current_int = np.mean(snap_image(calib.mmc))

            if current_int < min_int:
                better_lca = lca
                better_lcb = lcb
                min_int = current_int
                if calib.print_details:
                    print("update (%f, %f, %f)" % (min_int, better_lca, better_lcb))
                    
    if calib.print_details:
        print("coarse search done")
        print("better lca = " + str(better_lca))
        print("better lcb = " + str(better_lcb))
        print("better int = " + str(min_int))

    best_lca = better_lca
    best_lcb = better_lcb

    return best_lca, best_lcb, min_int

def optimize_minscalar(calib, reference, bound_range, mode, normalize=False):
    
    current_lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
    current_lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])
    
    # check that bounds don't exceed range of LC
    lca_lower_bound = 0.01 if (current_lca - bound_range) <= 0.01 else current_lca - bound_range
    lca_upper_bound = 1.6 if (current_lca + bound_range) >= 1.6 else current_lca + bound_range

    lcb_lower_bound = 0.01 if current_lcb - bound_range <= 0.01 else current_lcb - bound_range
    lcb_upper_bound = 1.6 if current_lcb + bound_range >= 1.6 else current_lcb + bound_range
    
    bounds = [(lca_lower_bound, lca_upper_bound),(lcb_lower_bound,lcb_upper_bound)]
    
    if mode == '60' or mode == '120':
        res = optimize.minimize_scalar(calib.opt_lc_cons, bounds=bounds[0], method='bounded', args=(reference,mode))

        # Brent Optimization
        set_lc(calib.mmc, res.x, calib.PROPERTIES['LCA'])
        
        swing = (calib.lca_ext - res.x) * calib.ratio
        
        if mode == '60':
            lcb = calib.lcb_ext + swing
            set_lc(calib.mmc, lcb, calib.PROPERTIES['LCB'])
        
        if mode == '120':
            lcb = calib.lcb_ext - swing
            set_lc(calib.mmc, lcb, calib.PROPERTIES['LCB'])
        
        difference = (res.fun / reference) * 100
        
        if calib.print_details:
            print('\tOptimizing lca w/ constrained lcb ...')
            print(f"\tlca = {res.x:.4f}")
            print(f'\tlcb = {lcb:.4f}')
            print(f'\tIntensity = {res.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%')
        
        return res.x, lcb, res.fun

    elif mode == '90':
        
        res = optimize.minimize_scalar(calib.opt_lc, bounds=bounds[0], method='bounded',args=(calib.PROPERTIES['LCA'],reference,normalize))
        
        set_lc(calib.mmc, res.x, calib.PROPERTIES['LCA'])
        
        difference = (res.fun / reference) * 100
        lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])
        if calib.print_details:
            print('\tOptimize lca ...')
            print(f"\tlca = {res.x:.4f}")
            print(f"\tlcb = {lcb:.4f}")
            print(f'\tIntensity = {res.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%')
            
            
        return res.x, lcb, res.fun
    
    elif mode == '45' or mode == '135':
        res = optimize.minimize_scalar(calib.opt_lc, bounds=bounds[1], method='bounded',args=(calib.PROPERTIES['LCB'],reference,normalize))
        
        set_lc(calib.mmc, res.x, calib.PROPERTIES['LCB'])
        
        difference = (res.fun / reference) * 100
        lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
        
        if calib.print_details:
            print('\tOptimize lcb ...')
            print(f"\tlca = {lca:.4f}")
            print(f"\tlcb = {res.x:.4f}")
            print(f'\tIntensity = {res.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%')
            
        return lca, res.x, res.fun
            
            
    else:
            
        optimal = []
        
        res_a=optimize.minimize_scalar(calib.opt_lc, bounds=bounds[0], method='bounded',args=(calib.PROPERTIES['LCA'],reference,normalize))
        
        set_lc(calib.mmc, res_a.x, calib.PROPERTIES['LCA'])
        
        lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])
        
        optimal.append([res_a.x, lcb, abs(res_a.fun)])
        
        difference = (res_a.fun / reference) * 100
        
        if calib.print_details:
            print('\tOptimize lca ...')
            print(f"\tlca = {res_a.x:.4f}")
            print(f"\tlcb = {lcb:.4f}")
            print(f'\tIntensity = {res_a.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%\n')

        res_b=optimize.minimize_scalar(calib.opt_lc, bounds=bounds[1], method='bounded',args=(calib.PROPERTIES['LCB'],reference,normalize))

        set_lc(calib.mmc, res_b.x, calib.PROPERTIES['LCB'])
        
        lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
        
        optimal.append([lca, res_b.x, abs(res_b.fun)])
        
        difference = (res_b.fun / reference) * 100
        
        if calib.print_details:
            print('\tOptimize lcb ...')
            print(f"\tlca = {lca:.4f}")
            print(f"\tlcb = {res_b.x:.4f}")
            print(f'\tIntensity = {res_b.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%\n')
            
            
            print(f'\tBegin Finer Search\n')
            
            
        #============BEGIN FINE SEARCH=================
            
        bounds = [(lca-.01, lca+.01), (res_b.x -.01, res_b.x + .01)]
        
        res_a=optimize.minimize_scalar(calib.opt_lc, bounds=bounds[0], method='bounded',args=(calib.PROPERTIES['LCA'],reference,normalize))
        
        set_lc(calib.mmc, res_a.x, calib.PROPERTIES['LCA'])
        
        lcb = get_lc(calib.mmc, calib.PROPERTIES['LCB'])
        
        optimal.append([res_a.x, lcb, abs(res_a.fun)])
        
        difference = (res_a.fun / reference) * 100
        
        if calib.print_details:
            print('\tOptimize lca ...')
            print(f"\tlca = {res_a.x:.4f}")
            print(f"\tlcb = {lcb:.4f}")
            print(f'\tIntensity = {res_a.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%\n')

        res_b=optimize.minimize_scalar(calib.opt_lc, bounds=bounds[1], method='bounded',args=(calib.PROPERTIES['LCB'],reference,normalize))

        set_lc(calib.mmc, res_b.x, calib.PROPERTIES['LCB'])
        
        lca = get_lc(calib.mmc, calib.PROPERTIES['LCA'])
        
        optimal.append([lca, res_b.x, abs(res_b.fun)])
        
        difference = (res_b.fun / reference) * 100
        
        optimal = np.asarray(optimal)
#         print(optimal)
        opt = np.where(optimal == np.min(optimal[:][2]))[0]
        
#         print(optimal[opt])
        set_lc(calib.mmc, float(optimal[opt][0][0]), calib.PROPERTIES['LCA'])
        set_lc(calib.mmc, float(optimal[opt][0][1]), calib.PROPERTIES['LCB'])
            
        if calib.print_details:
            print('\tOptimize lcb ...')
            print(f"\tlca = {lca:.4f}")
            print(f"\tlcb = {res_b.x:.4f}")
            print(f'\tIntensity = {res_b.fun + reference}')
            print(f'\tIntensity Difference = {difference:.7f}%')
            
            print(f'\n Lowest Intensity: {optimal[opt][0][2]:.4f}, lca = {optimal[opt][0][0]:.4f}, lcb = {optimal[opt][0][1]:.7f}')
            
            
            
        return lca, res_b.x, res_b.fun