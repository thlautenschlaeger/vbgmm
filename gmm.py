import numpy as np
from scipy.special import digamma
from scipy.special import gamma
from scipy.special import loggamma


class VBGMM:
    """ Class providing a framework for Variational Gaussian Mixture Models.
    Use method optimize to perform an optimization step."""

    def __init__(self, x, k, alpha_0, beta_0, m_0, W_0, nu_0, normalize=False):
        """
        :param x: input data
        :param k: number mixture components
        :param alpha_0: hyper parameter for dirichlet prior
        :param beta_0: scaling factor for sampled covariance matrix
        :param m_0: mean but usually set to 0
        :param W_0: Wishart scale matrix
        :param nu_0: degrees of freedom for Wishart distribution
        :param normalize: normalizes the input data if set to true
        """

        self.x = x[:, :, np.newaxis]
        self.k = k
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.m_0 = m_0
        self.W_0 = W_0
        self.nu_0 = nu_0
        self.dim = x.shape[1]
        self.n = x.shape[0]

        # initialize params
        self.alpha_k = np.abs(np.random.standard_normal(size=self.k)) / 10
        self.beta_k = np.abs(np.random.standard_normal(size=self.k))
        self.m_k = np.random.standard_normal(size=(self.k, self.dim, 1))
        self.w_k = np.random.standard_normal(size=(self.k, self.dim, self.dim))
        for k in range(self.k):
            self.w_k[k] = self.w_k[k] @ self.w_k[k].T

        # self.w_k = 10 # scaling covariance matrix

        self.nu_k = np.abs(np.random.standard_normal(size=self.k))

        # create variable memory
        self.log_lamb = np.empty(shape=self.k)
        self.log_pi = np.empty(shape=self.k)
        self.estimate = np.empty(shape=(self.n, self.k))

        self.rho = np.empty(shape=(self.n, self.k))
        self.resp = np.empty(shape=(self.n, self.k))

        self.counts = np.empty(shape=self.k)
        self.means = np.empty(shape=(self.k, self.dim, 1))
        self.covars = np.empty(shape=(self.k, self.dim, self.dim))

        if normalize:
            self.normalize()

    def normalize(self):
        mean = np.mean(self.x, axis=0)
        stdev = np.std(self.x, axis=0)
        self.x = (self.x - mean) / stdev

    def vbe_step(self):
        """ Variational e-step - Computes expectations of the precision matrix lambda and the
         mixture coefficients pi """

        digam_alpha = digamma(np.sum(self.alpha_k))
        for k in range(self.k):

            # compute estimate over ln det(lamb)
            tmp = sum(digamma((self.nu_k[k] + 1 - j) / 2) for j in range(self.dim))

            det = np.linalg.det(self.w_k[k])
            self.log_lamb[k] = tmp + self.dim * np.log(2) + np.log(det)

            # compute estimate for ln pi
            self.log_pi[k] = digamma(self.alpha_k[k]) - digam_alpha

            for n in range(self.n):
                tmp = self.x[n] - self.m_k[k]
                # compute estimate over mu and lambda
                self.estimate[n, k] = self.dim * (1 / self.beta_k[k]) + self.nu_k[k] * (tmp.T @ self.w_k[k] @ tmp)

    def compute_responsibilities(self):
        for n in range(self.n):
            for k in range(self.k):
                tmp = self.x[n] - self.m_k[k]
                tmp1 = np.exp(- (self.dim / (2 * self.beta_k[k])) - (self.nu_k[k] / 2) * tmp.T @ self.w_k[k] @ tmp)
                self.rho[n, k] = (np.exp(self.log_pi[k]) * (np.exp(self.log_lamb[k]) ** 0.5)) * tmp1

        normalize_term = np.nan_to_num(np.sum(self.rho, axis=1))
        for n in range(self.n):
            for k in range(self.k):
                self.resp[n, k] = np.nan_to_num(self.rho[n, k] / normalize_term[n])


    def compute_sufficient_stats(self):
        """ Computes the sufficient GMM statistics """
        self.counts = (np.sum(self.resp, axis=0) + 10e-30)
        # print(self.counts)
        for k in range(self.k):
            self.means[k] = np.sum(self.resp[n, k] * self.x[n] for n in range(self.n)) / self.counts[k]
            self.covars[k] = np.sum(self.resp[n, k] * (self.x[n] - self.means[k]) @ (self.x[n] - self.means[k]).T
                                    for n in range(self.n)) / self.counts[k]
            self.covars[k] = np.nan_to_num(self.covars[k])
            self.means[k] = np.nan_to_num(self.means[k])

    def vbmstep(self):
        """ Computes variational bayesian m-step """
        for k in range(self.k):
            self.beta_k[k] = self.beta_0 + self.counts[k]
            self.m_k[k] = (1 / self.beta_k[k]) * (self.beta_0 * self.m_0 +
                                                  self.counts[k] * self.means[k])

            tmp = (self.beta_0 * self.counts[k]) / (self.beta_0 + self.counts[k])
            tmp2 = (self.means[k] - self.m_0)
            tmp = np.linalg.inv(self.W_0) + self.counts[k] * self.covars[k] + tmp * tmp2 @ tmp2.T
            self.w_k[k] = np.linalg.inv(tmp)
            self.nu_k[k] = self.nu_0 + self.counts[k]
            self.alpha_k[k] = self.alpha_0[k] + self.counts[k]

    def optimize(self):
        """ Performs one optimization iteration of VBGMM """
        self.vbe_step()
        self.compute_responsibilities()
        self.compute_sufficient_stats()
        self.vbmstep()

    def compute_vlb(self):
        # estimate E[ln p(X|Z,mu, Lambda)]]
        estimate1 = 0
        for k in range(self.k):
            tmp1 = self.log_lamb[k] - self.dim * self.beta_k[k] ** (-1)
            tmp2 = -self.nu_k[k] * np.trace(self.covars[k] @ self.w_k[k])
            term2 = -self.nu_k[k] * ((self.means[k] - self.m_k[k]).T @ self.w_k[k] @ (self.means[k] - self.m_k[k])) \
                    - self.dim * np.log(2 * np.pi)
            estimate1 += self.counts[k] * (tmp1 + tmp2 + term2)
        estimate1 *= 0.5

        # estimate E[ln p(Z|pi)]
        estimate2 = 0
        for n in range(self.n):
            for k in range(self.k):
                estimate2 += self.resp[n, k] * self.log_pi[k]

        # estimate E[ln p(pi)]
        estimate3 = self.ln_C(self.alpha_0) + (self.alpha_0[0] - 1) * np.sum(self.log_pi)

        # estimate E[ln p(mu, Lambda)]]
        term1 = 0
        term4 = 0
        for k in range(self.k):
            tmp1 = self.dim * np.log(self.beta_0 / (2 * np.pi)) + self.log_lamb[k] - (self.dim * self.beta_0) / \
                   self.beta_k[k]
            tmp2 = -self.beta_0 * self.nu_k[k] * ((self.m_k[k] - self.m_0).T @ self.w_k[k] @ (self.m_k[k] - self.m_0))

            term1 += tmp1 + tmp2

            term4 += self.nu_k[k] * np.trace(np.linalg.inv(self.W_0) @ self.w_k[k])

        term1 *= 0.5
        term4 *= 0.5
        term2 = self.k * self.ln_B(self.W_0, self.nu_0)
        term3 = ((self.nu_0 - self.dim - 1) / 2) * np.sum(self.log_lamb)

        estimate4 = term1 + term2 + term3 - term4

        # estimate E[ln q(Z)])]
        estimate5 = 0
        for n in range(self.n):
            for k in range(self.k):
                estimate5 += np.nan_to_num(self.resp[n, k] * (np.log(self.resp[n, k])))
                # print( "log: {} | resp: {}".format(np.log(self.resp[n, k]), self.resp[n, k]))

        # estimate E[ln q(pi)]
        estimate6 = 0
        for k in range(self.k):
            estimate6 += (self.alpha_k[k] - 1) * self.log_pi[k] + (self.ln_C(self.alpha_k))

        # estimate E[ln q(mu, Lambda)]
        estimate7 = 0
        for k in range(self.k):
            term1 = 0.5 * self.log_lamb[k] + (self.dim / 2) * np.log(self.beta_k[k] / (2 * np.pi))
            term2 = self.dim / 2
            term3 = self.H(self.w_k[k], self.nu_k[k], self.log_lamb[k])
            estimate7 += term1 - term2 - term3

        elbo = estimate1 + estimate2 + estimate3 + estimate4 - estimate5 - estimate6 - estimate7

        return elbo

    def ln_C(self, alpha):
        num = loggamma(np.sum(alpha))
        den = np.sum([loggamma(alpha[k]) for k in range(self.k)])

        return num - den

    def C(self, alpha):
        num = gamma(np.sum(alpha))
        den = np.sum([gamma(alpha[k]) for k in range(self.k)])

        return num / den

    def B(self, W, nu):
        dim = W.shape[0]
        term1 = np.linalg.det(W) ** (-nu / 2)
        term2 = 2 ** (nu * dim / 2) * np.pi ** (dim * (dim - 1) / 4)
        term3 = np.prod([gamma(nu + 1 - i) for i in range(1, dim + 1)])
        term4 = (term2 * term3) ** (-1)

        return term1 * term4

    def ln_B(self, W, nu):
        dim = W.shape[0]
        term1 = np.log(np.linalg.det(W)) * (-nu / 2)
        term2 = np.log(2) * (nu * dim / 2) + np.log(np.pi) * (dim * (dim - 1) / 4)
        term3 = np.sum([loggamma(nu + 1 - i) for i in range(1, dim + 1)])
        term4 = (term2 + term3) * (-1)

        return term1 + term4

    def H(self, W, nu, lamb):
        dim = W.shape[0]
        term1 = -self.ln_B(W, nu)
        term2 = ((nu - dim - 1) / 2) * lamb + (nu * dim) / 2

        return term1 - term2
