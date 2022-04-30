import numpy as np
from scipy import optimize
from recOrder.io.core_functions import snap_and_average
import logging

class BrentOptimizer:

    def __init__(self, calib):

        self.calib = calib


    def _check_bounds(self, lca_bound, lcb_bound):

        current_lca = self.calib.get_lc('LCA')
        current_lcb = self.calib.get_lc('LCB')

        # check that bounds don't exceed range of LC
        lca_lower_bound = 0.01 if (current_lca - lca_bound) <= 0.01 else current_lca - lca_bound
        lca_upper_bound = 1.6 if (current_lca + lca_bound) >= 1.6 else current_lca + lca_bound

        lcb_lower_bound = 0.01 if current_lcb - lcb_bound <= 0.01 else current_lcb - lcb_bound
        lcb_upper_bound = 1.6 if current_lcb + lcb_bound >= 1.6 else current_lcb + lcb_bound

        return lca_lower_bound, lca_upper_bound, lcb_lower_bound, lcb_upper_bound

    def opt_lca(self, cost_function, lower_bound, upper_bound, reference, cost_function_args):

        xopt, fval, ierr, numfunc = optimize.fminbound(cost_function,
                                                       x1=lower_bound,
                                                       x2=upper_bound,
                                                       disp=0,
                                                       args=cost_function_args,
                                                       full_output=True)

        lca = xopt
        lcb = self.calib.get_lc(self.calib.mmc, self.calib.PROPERTIES['LCA'])
        abs_intensity = fval + reference
        difference = fval / reference * 100

        logging.debug('\tOptimize lca ...')
        logging.debug(f"\tlca = {lca:.5f}")
        logging.debug(f"\tlcb = {lcb:.5f}")
        logging.debug(f'\tIntensity = {abs_intensity}')
        logging.debug(f'\tIntensity Difference = {difference:.4f}%')

        return [lca, lcb, abs_intensity, difference]

    def opt_lcb(self, cost_function, lower_bound, upper_bound, reference, cost_function_args):

        xopt, fval, ierr, numfunc = optimize.fminbound(cost_function,
                                                       x1=lower_bound,
                                                       x2=upper_bound,
                                                       disp=0,
                                                       args=cost_function_args,
                                                       full_output=True)

        lca = self.calib.get_lc(self.calib.mmc, self.calib.PROPERTIES['LCA'])
        lcb = xopt
        abs_intensity = fval + reference
        difference = fval / reference * 100

        logging.debug('\tOptimize lcb ...')
        logging.debug(f"\tlca = {lca:.5f}")
        logging.debug(f"\tlcb = {lcb:.5f}")
        logging.debug(f'\tIntensity = {abs_intensity}')
        logging.debug(f'\tIntensity Difference = {difference:.4f}%')

        return [lca, lcb, abs_intensity, difference]

    def optimize(self, state, lca_bound, lcb_bound, reference, thresh, n_iter):

        converged = False
        iteration = 1
        self.calib.inten = []
        optimal = []

        while not converged:
            logging.debug(f'iteration: {iteration}')

            lca_lower_bound, lca_upper_bound,\
            lcb_lower_bound, lcb_upper_bound = self._check_bounds(lca_bound, lcb_bound)

            if state == 'ext':

                results_lca = self.opt_lca(self.calib.opt_lc, lca_lower_bound, lca_upper_bound,
                                            reference, (self.calib.PROPERTIES['LCA'], reference))

                self.calib.set_lc(self.calib.mmc, results_lca[0], 'LCA')

                optimal.append(results_lca)

                results_lcb = self.opt_lcb(self.calib.opt_lc, lcb_lower_bound, lcb_upper_bound,
                                            reference, (self.calib.PROPERTIES['LCB'], reference))

                self.calib.set_lc(self.calib.mmc, results_lca[1], 'LCB')

                optimal.append(results_lcb)

                results = results_lcb

            if state == '45' or state == '135':

                results = self.opt_lcb(self.calib.opt_lc, lca_lower_bound, lca_upper_bound,
                                        reference, (self.calib.PROPERTIES['LCB'], reference))

                optimal.append(results)

            if state == '60':

                results = self.opt_lca(self.calib.opt_lc_cons, lca_lower_bound, lca_upper_bound,
                                        reference, (reference, '60'))

                swing = (self.calib.lca_ext - results[0]) * self.calib.ratio
                lca = results[0]
                lcb = self.calib.lcb_ext + swing

                optimal.append([lca, lcb, results[2], results[3]])

            if state == '90':

                results = self.opt_lca(self.calib.opt_lc, lca_lower_bound, lca_upper_bound,
                                           reference, (self.calib.PROPERTIES['LCA'], reference))

                optimal.append(results)

            if state == '120':
                results = self.opt_lca(self.calib.opt_lc_cons, lca_lower_bound, lca_upper_bound,
                                       reference, (reference, '120'))

                swing = (self.calib.lca_ext - results[0]) * self.calib.ratio
                lca = results[0]
                lcb = self.calib.lcb_ext - swing

                optimal.append([lca, lcb, results[2], results[3]])

            # if both LCA and LCB meet threshold, stop
            if results[3] <= thresh:
                converged = True
                optimal = np.asarray(optimal)

                return optimal[-1, 0], optimal[-1, 1], optimal[-1, 2]

            # if loop preforms more than n_iter iterations, stop
            elif iteration >= n_iter:
                logging.debug(f'Exceeded {n_iter} Iterations: Search discontinuing')

                converged = True
                optimal = np.asarray(optimal)
                opt = np.where(optimal == np.min(np.abs(optimal[:, 0])))[0]

                logging.debug(f'Lowest Inten: {optimal[opt, 0]}, lca = {optimal[opt, 1]}, lcb = {optimal[opt, 2]}')

                return optimal[-1, 0], optimal[-1, 1], optimal[-1, 2]

            iteration += 1


class MinScalarOptimizer:

    def __init__(self, calib):

        self.calib = calib

    def _check_bounds(self, lca_bound, lcb_bound):

        current_lca = self.calib.get_lc('LCA')
        current_lcb = self.calib.get_lc('LCB')

        if self.calib.mode == 'voltage':
            # check that bounds don't exceed range of LC
            lca_lower_bound = 0.01 if (current_lca - lca_bound) <= 0.01 else current_lca - lca_bound
            lca_upper_bound = 2.2 if (current_lca + lca_bound) >= 2.2 else current_lca + lca_bound

            lcb_lower_bound = 0.01 if current_lcb - lcb_bound <= 0.01 else current_lcb - lcb_bound
            lcb_upper_bound = 2.2 if current_lcb + lcb_bound >= 2.2 else current_lcb + lcb_bound

        else:
            # check that bounds don't exceed range of LC
            lca_lower_bound = 0.01 if (current_lca - lca_bound) <= 0.01 else current_lca - lca_bound
            lca_upper_bound = 1.6 if (current_lca + lca_bound) >= 1.6 else current_lca + lca_bound

            lcb_lower_bound = 0.01 if current_lcb - lcb_bound <= 0.01 else current_lcb - lcb_bound
            lcb_upper_bound = 1.6 if current_lcb + lcb_bound >= 1.6 else current_lcb + lcb_bound

        return lca_lower_bound, lca_upper_bound, lcb_lower_bound, lcb_upper_bound

    def opt_lca(self, cost_function, lower_bound, upper_bound, reference, cost_function_args):

        res = optimize.minimize_scalar(cost_function, bounds=(lower_bound, upper_bound),
                                       method='bounded', args=cost_function_args)

        lca = res.x
        lcb = self.calib.get_lc('LCB')
        abs_intensity = res.fun + reference
        difference = res.fun / reference * 100

        logging.debug('\tOptimize lca ...')
        logging.debug(f"\tlca = {lca:.5f}")
        logging.debug(f"\tlcb = {lcb:.5f}")
        logging.debug(f'\tIntensity = {abs_intensity}')
        logging.debug(f'\tIntensity Difference = {difference:.4f}%')

        return [lca, lcb, abs_intensity, difference]

    def opt_lcb(self, cost_function, lower_bound, upper_bound, reference, cost_function_args):

        res = optimize.minimize_scalar(cost_function, bounds=(lower_bound, upper_bound),
                                       method='bounded', args=cost_function_args)

        lca = self.calib.get_lc('LCA')
        lcb = res.x
        abs_intensity = res.fun + reference
        difference = res.fun / reference * 100

        logging.debug('\tOptimize lcb ...')
        logging.debug(f"\tlca = {lca:.5f}")
        logging.debug(f"\tlcb = {lcb:.5f}")
        logging.debug(f'\tIntensity = {abs_intensity}')
        logging.debug(f'\tIntensity Difference = {difference:.4f}%')

        return [lca, lcb, abs_intensity, difference]

    def optimize(self, state, lca_bound, lcb_bound, reference, thresh=None, n_iter=None):

        lca_lower_bound, lca_upper_bound, lcb_lower_bound, lcb_upper_bound = self._check_bounds(lca_bound, lcb_bound)

        if state == 'ext':
            optimal = []

            results_lca = self.opt_lca(self.calib.opt_lc, lca_lower_bound, lca_upper_bound,
                                       reference, ('LCA', reference))

            self.calib.set_lc(results_lca[0], 'LCA')

            optimal.append(results_lca)

            results_lcb = self.opt_lcb(self.calib.opt_lc, lcb_lower_bound, lcb_upper_bound,
                                       reference, ('LCB', reference))

            self.calib.set_lc(results_lcb[1], 'LCB')

            optimal.append(results_lcb)

            # ============BEGIN FINE SEARCH=================

            logging.debug(f'\n\tBeginning Finer Search\n')
            lca_lower_bound = results_lcb[0] - .01
            lca_upper_bound = results_lcb[0] + .01
            lcb_lower_bound = results_lcb[1] - .01
            lcb_upper_bound = results_lcb[1] + .01

            results_lca = self.opt_lca(self.calib.opt_lc, lca_lower_bound, lca_upper_bound,
                                       reference, ('LCA', reference))

            self.calib.set_lc(results_lca[0], 'LCA')

            optimal.append(results_lca)

            results_lcb = self.opt_lcb(self.calib.opt_lc, lcb_lower_bound, lcb_upper_bound,
                                       reference, ('LCB', reference))

            self.calib.set_lc(results_lcb[1], 'LCB')

            optimal.append(results_lcb)

            # Sometimes this optimization can drift away from the minimum,
            # this makes sure we use the lowest iteration
            optimal = np.asarray(optimal)
            opt = np.where(optimal == np.min(optimal[:][2]))[0]

            lca = float(optimal[opt][0][0])
            lcb = float(optimal[opt][0][1])
            results = optimal[opt][0]

        if state == '45' or state == '135':
            results = self.opt_lcb(self.calib.opt_lc, lcb_lower_bound, lcb_upper_bound,
                                   reference, ('LCB', reference))

            lca = results[0]
            lcb = results[1]

        if state == '60':
            results = self.opt_lca(self.calib.opt_lc_cons, lca_lower_bound, lca_upper_bound,
                                   reference, ('LCA', reference, '60'))

            swing = (self.calib.lca_ext - results[0]) * self.calib.ratio
            lca = results[0]
            lcb = self.calib.lcb_ext + swing

        if state == '90':
            results = self.opt_lca(self.calib.opt_lc, lca_lower_bound, lca_upper_bound,
                                       reference, ('LCA', reference))
            lca = results[0]
            lcb = results[1]

        if state == '120':
            results = self.opt_lca(self.calib.opt_lc_cons, lca_lower_bound, lca_upper_bound,
                                   reference, ('LCB', reference, '120'))

            swing = (self.calib.lca_ext - results[0]) * self.calib.ratio
            lca = results[0]
            lcb = self.calib.lcb_ext - swing

        return lca, lcb, results[2]
