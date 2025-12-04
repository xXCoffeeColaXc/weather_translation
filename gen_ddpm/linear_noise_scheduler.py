import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """

    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.one_minus_cum_prod = 1 - self.alpha_cum_prod
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        self.max_sigma = self.sqrt_one_minus_alpha_cum_prod.max()

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cum_prod = self.alpha_cum_prod.to(device)
        self.sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(device)
        self.one_minus_cum_prod = self.one_minus_cum_prod.to(device)
        self.sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)
        self.max_sigma = self.max_sigma.to(device)

    def add_noise2(self, original, noise, t):
        device = original.device
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(device)[t]
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)[t]

        view_shape = (-1,) + (1,) * (original.dim() - 1)
        sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.view(view_shape)
        sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.view(view_shape)

        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original +
                sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def sample_prev_timestep2(self, xt, noise_pred, t) -> tuple[torch.Tensor, torch.Tensor]:
        device = xt.device
        beta = self.betas.to(device)[t].view(-1, 1, 1, 1)
        alpha = self.alphas.to(device)[t].view(-1, 1, 1, 1)
        alpha_cum = self.alpha_cum_prod.to(device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)[t].view(-1, 1, 1, 1)

        mean = (xt - (beta * noise_pred) / (sqrt_one_minus_alpha_cum_prod)) / torch.sqrt(alpha)

        if torch.all(t == 0):
            sigma = torch.zeros_like(xt)
            return mean, sigma

        prev_t = (t - 1).clamp(min=0)
        alpha_cum_prev = self.alpha_cum_prod.to(device)[prev_t].view(-1, 1, 1, 1)
        posterior_variance = ((1 - alpha_cum_prev) / (1 - alpha_cum)) * beta
        sigma = torch.sqrt(posterior_variance)
        z = torch.randn_like(xt)
        return mean, sigma * z

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        #x0 = xt - sqrt(1 - alpha) * noise_pred / sqrt(alpha)
        # x0 = (
        #     (xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
        #     torch.sqrt(self.alpha_cum_prod.to(xt.device)[t])
        # )
        # x0 = torch.clamp(x0, -1., 1.)

        # mean = 1/sqrt(alpha) * (xt - (1-alpha)/(sqrt(1-alpha_hat)) * noise_pred)
        mean = xt - (((self.betas.to(xt.device)[t]) * noise_pred) /
                     (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t]))

        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, None, None
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(xt.device)

            # OR
            # variance = self.betas[t]
            # sigma = variance**0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean, sigma * z, None
