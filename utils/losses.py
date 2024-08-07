import torch
import torch.nn.functional as F
import numpy as np
import math

def GradNorm(disp):
    dy = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
    dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    dxx = torch.abs(dx[:, :, :, 1:] - dx[:, :,  :, :-1])
    dyy = torch.abs(dy[:, :, 1:, :] - dy[:, :,  :-1, :])
    d = (torch.mean(dxx) +  torch.mean(dyy)) / 2
    return d

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image=None, with_img=False, alpha=10):
    if with_img:
        img_dx, img_dy = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

        dx, dy = gradient(flo)

        loss_x = weights_x * dx.abs() / 2.
        loss_y = weights_y * dy.abs() / 2

        return loss_x.mean() / 2. + loss_y.mean() / 2.
    else:
        dx, dy = gradient(flo)
        return dx.abs().mean() / 2. + dy.abs().mean() / 2.
        
def smooth_grad_2nd(flo, image=None, with_img=True, alpha=10):
    if with_img:
        img_dx, img_dy = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

        dx, dy = gradient(flo)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)

        loss_x = weights_x[:, :, :, 1:] * dx2.abs()
        loss_y = weights_y[:, :, 1:, :] * dy2.abs()
        loss_xy = weights_y[:, :, 1:, :] * dxdy2.abs()

        return loss_x.abs().mean() + loss_y.abs().mean() + loss_xy.abs().mean()
    else:
        dx, dy = gradient(flo)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return dx2.abs().mean() + dy2.abs().mean() + dxdy.abs().mean()

def LNCC(pred, target, win = [160,8]):
    # Unfold pred and target into patches
    unfold = torch.nn.Unfold(kernel_size=(win[0],win[1]), stride = (win[0],win[1]))
    pred = unfold(pred).permute(0,2,1)
    target = unfold(target).permute(0,2,1)
    target_mean = torch.mean(target, dim=-1, keepdim=True)
    target_std = torch.std(target, dim=-1)
    pred_mean = torch.mean(pred, dim=-1, keepdim=True)
    pred_std = torch.std(pred, dim=-1)
    ncc = torch.sum((target - target_mean) * (pred - pred_mean),dim=-1)  / (win[0]*win[1]*target_std*pred_std+1e-18)
    return torch.mean(ncc)

def PhotoLoss(pre,post,photo_loss_type='charbonnier',photo_loss_delta=1.5):
    if photo_loss_type == 'abs_robust':
        photo_diff = pre - post
        loss_diff = (torch.abs(photo_diff) + 0.01).pow(photo_loss_delta)
    elif photo_loss_type == 'charbonnier':
        photo_diff = pre - post
        loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
    elif photo_loss_type == 'L1':
        photo_diff = pre - post
        loss_diff = torch.abs(photo_diff + 1e-6)
    return loss_diff.mean()


def PDE(disp):
    u_x = torch.pow(disp[:,:1,:,1:] - disp[:,:1,:,:-1], 2)
    u_y = torch.pow(disp[:,:1,1:,:] - disp[:,:1,:-1,:], 2)

    v_y = torch.pow(disp[:,1:,1:,:]-disp[:,1:,:-1,:], 2)
    v_x = torch.pow(disp[:,1:,:,1:]-disp[:,1:,:,:-1], 2)

    pde = 2*u_x.mean() + u_y.mean() + 2*v_x.mean() + v_y.mean()

    # [b,c,h,w] = disp.shape
    # dy = disp[:,:,0:h-2,:] + disp[:,:,2:h,:] - 2*disp[:,:,1:h-1,:]
    # dx = disp[:,:,:,0:w-2] + disp[:,:,:,2:w] - 2 * disp[:,:,:,1:w-1]
    # pde = dy.abs().mean() + dx.abs().mean()


    # u_xx = torch.abs(disp[:,:1,2:h-2,1:w-3]+disp[:,:1,2:h-2,3:w-1]-2*disp[:,:1,2:h-2,2:w-2]).mean()
    # v_xy = torch.abs(disp[:,1:,3:h-1,3:w-1]+disp[:,1:,1:h-3,1:w-3]-disp[:,1:,1:h-3,3:w-1]-disp[:,1:,3:h-1,1:w-3]).mean()
    # u_x4 = torch.abs(disp[:,:1,2:h-2,4:w]-4*disp[:,:1,2:h-2,3:w-1]+6*disp[:,:1,2:h-2,2:w-2]-4*disp[:,:1,2:h-2,1:w-3]+disp[:,:1,2:h-2,0:w-4]).mean()
    # v_x2y2 = torch.abs((disp[:,1:,3:h-1,3:w-1]+disp[:,1:,3:h-1,1:w-3]-2*disp[:,1:,3:h-1,2:w-2])\
    #         +(disp[:,1:,1:h-3,1:w-3]+disp[:,1:,1:h-3,3:w-1]-2*disp[:,1:,1:h-3,2:w-2])\
    #         -2*(disp[:,1:,2:h-2,1:w-3]+disp[:,1:,2:h-2,3:w-1]-2*disp[:,1:,2:h-2,2:w-2])).mean()

    # v_yy = torch.abs(disp[:,1:,1:h-3,2:w-2]+disp[:,1:,3:h-1,2:w-2]-2*disp[:,1:,2:h-2,2:w-2]).mean()
    # u_xy = torch.abs(disp[:,:1,3:h-1,3:w-1]+disp[:,:1,1:h-3,1:w-3]-disp[:,:1,1:h-3,3:w-1]-disp[:,:1,3:h-1,1:w-3]).mean()
    # v_y4 = torch.abs(disp[:,1:,4:h,2:w-2]-4*disp[:,1:,3:h-1,2:w-2]+6*disp[:,1:,2:h-2,2:w-2]-4*disp[:,1:,1:h-3,2:w-2]+disp[:,1:,0:h-4,2:w-2]).mean()
    # u_x2y2 = torch.abs((disp[:,:1,3:h-1,3:w-1]+disp[:,:1,3:h-1,1:w-3]-2*disp[:,:1,3:h-1,2:w-2])\
    #         +(disp[:,:1,1:h-3,1:w-3]+disp[:,:1,1:h-3,3:w-1]-2*disp[:,:1,1:h-3,2:w-2])\
    #         -2*(disp[:,:1,2:h-2,1:w-3]+disp[:,:1,2:h-2,3:w-1]-2*disp[:,:1,2:h-2,2:w-2])).mean()

    # pde = (u_xx+v_xy) / 2  + (v_yy+u_xy) / 2

    return pde


def NCC(pred, target):
    size_target_image = torch.numel(target)
    target_mean = torch.mean(target)
    target_std = torch.std(target)
    pred_mean = torch.mean(pred)
    pred_std = torch.std(pred)
    ncc = torch.sum((target -target_mean) * (pred - pred_mean)) / (size_target_image*target_std*pred_std+1e-18)
    return ncc

def LNCC_voxelmorph(y_pred, y_true, win=[9, 9]):
    """
    Local (over window) normalized cross correlation loss.(voxelmorph pytorch)
    """
    I = y_true
    J = y_pred
    channel_size = I.size()[1]
    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(I.size())) - 2
    assert ndims in [
        1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # compute filters
    sum_filt = torch.ones([1, channel_size, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = (cross * cross) / (I_var * J_var + 1e-7)
    return torch.mean(cc) 

def RMSE(pred,target):
    target = -target
    pred = -pred
    return torch.sqrt(torch.mean((target-pred)**2))

def SNRe(strain):
    return np.mean(strain) / np.std(strain)

def NRMSE(pred,target):
    target = -target
    pred = -pred
    return (torch.sqrt(torch.mean((target-pred)**2))/torch.mean(target))*100


# def PDE(disp,kernel_size=7):
    # input:disp (1,2,2048,256)

    # axial_disp = torch.squeeze(disp[:, 1, :, :]) # org_v
    # lateral_disp = torch.squeeze(disp[:, 0, :, :]) # org_u
    # (h,w) = axial_disp.shape

    # step_1:Pretreatment 预处理
    # error_range = 0.25
    # med_disp = median_blur(disp,(kernel_size,kernel_size))
    # med_axial_disp = torch.squeeze(med_disp[:, 1, :, :])
    # med_lateral_disp = torch.squeeze(med_disp[:, 0, :, :])
    # print(med_axial_disp[0,0])
    # print('med_lateral_disp.shape = ',med_lateral_disp.shape)
    # err_axial = torch.abs(axial_disp-med_axial_disp)
    # err_lateral = torch.abs(lateral_disp-med_lateral_disp)
    # idy = torch.where(err_axial > error_range)
    # idx = torch.where(err_lateral > error_range)
    # axial_disp[idy] = med_axial_disp[idy]
    # lateral_disp[idx] = med_lateral_disp[idx]

    # step_2:PDE
    # pre_u = lateral_disp.clone()
    # pre_v = axial_disp.clone()
    # post_u = torch.zeros_like(pre_u)
    # post_v = torch.zeros_like(pre_v)

    # dx = 0.1
    # dy = 0.1
    # cfl = 0.005
    # dt = cfl * pow(min(dx, dy), 2)
    # lamda_s = 0.2 # (elamda_0)
    # lamda_x = 0.01 # (elamda_1)
    # lamda_y = 20 # (elamda_2)

    # v1
    # u_xx = (pre_u[1:h-1,0:w-2]+pre_u[1:h-1,2:w]-2*pre_u[1:h-1,1:w-1]) / pow(dx,2)
    # v_xy = (pre_v[0:h-2,0:w-2]+pre_v[2:h,2:w]-pre_v[0:h-2,2:w]-pre_v[2:h,0:w-2])/(4*dx*dy)
    # hgu = u_xx + v_xy - lamda_x *(pre_u[1:h-1,1:w-1]-lateral_disp[1:h-1,1:w-1])
    # post_u[1:h-1,1:w-1] = pre_u[1:h-1,1:w-1] + dt*hgu

    # v_yy = (pre_v[0:h-2,1:w-1]+pre_v[2:h,1:w-1]-2*pre_v[1:h-1,1:w-1])/pow(dy,2)
    # u_xy = (pre_u[0:h-2,0:w-2]+pre_u[2:h,2:w]-pre_u[0:h-2,2:w]-pre_u[2:h,0:w-2])/(4*dx*dy)
    # hgv = v_yy + u_xy - lamda_y*(pre_v[1:h-1,1:w-1])
    # post_v[1:h-1,1:w-1] = pre_v[1:h-1,1:w-1] + dt*hgv

    # v2_smooth
    # 1st round
    # u_xx = (pre_u[2:h-2,1:w-3]+pre_u[2:h-2,3:w-1]-2*pre_u[2:h-2,2:w-2]) / pow(dx,2)
    # v_xy = (pre_v[3:h-1,3:w-1]+pre_v[1:h-3,1:w-3]-pre_v[1:h-3,3:w-1]-pre_v[3:h-1,1:w-3])/(4*dx*dy)
    # u_x4 = (pre_u[2:h-2,4:w]-4*pre_u[2:h-2,3:w-1]+6*pre_u[2:h-2,2:w-2]-4*pre_u[2:h-2,1:w-3]+pre_u[2:h-2,0:w-4])/pow(dx,4)
    # v_x2y2 = ((pre_v[3:h-1,3:w-1]+pre_v[3:h-1,1:w-3]-2*pre_v[3:h-1,2:w-2])\
    #         +(pre_v[1:h-3,1:w-3]+pre_v[1:h-3,3:w-1]-2*pre_v[1:h-3,2:w-2])\
    #         -2*(pre_v[2:h-2,1:w-3]+pre_v[2:h-2,3:w-1]-2*pre_v[2:h-2,2:w-2]))/(pow(dx,2)*pow(dy,2))
    # hgu = (u_xx+v_xy)-lamda_s*(u_x4+v_x2y2)-lamda_x*(pre_u[2:h-2,2:w-2]-lateral_disp[2:h-2,2:w-2])

    # v_yy = (pre_v[1:h-3,2:w-2]+pre_v[3:h-1,2:w-2]-2*pre_v[2:h-2,2:w-2])/pow(dy,2)
    # u_xy = (pre_u[3:h-1,3:w-1]+pre_u[1:h-3,1:w-3]-pre_u[1:h-3,3:w-1]-pre_u[3:h-1,1:w-3])/(4*dx*dy)
    # v_y4 = (pre_v[4:h,2:w-2]-4*pre_v[3:h-1,2:w-2]+6*pre_v[2:h-2,2:w-2]-4*pre_v[1:h-3,2:w-2]+pre_v[0:h-4,2:w-2])/pow(dy,4)
    # u_x2y2 = ((pre_u[3:h-1,3:w-1]+pre_u[3:h-1,1:w-3]-2*pre_u[3:h-1,2:w-2])\
    #         +(pre_u[1:h-3,1:w-3]+pre_u[1:h-3,3:w-1]-2*pre_u[1:h-3,2:w-2])\
    #         -2*(pre_u[2:h-2,1:w-3]+pre_u[2:h-2,3:w-1]-2*pre_u[2:h-2,2:w-2]))/(pow(dx,2)*pow(dy,2))
    # hgv = (u_xy+v_yy)-lamda_s*(u_x2y2+v_y4)-lamda_y*(pre_v[2:h-2,2:w-2]-axial_disp[2:h-2,2:w-2])

    # post_u[2:h-2,2:w-2] = pre_u[2:h-2,2:w-2]+dt*hgu
    # post_v[2:h-2,2:w-2] = pre_v[2:h-2,2:w-2]+dt*hgv

    # post_u[0,:] = post_u[4,:]
    # post_u[1,:] = post_u[3,:]
    # post_u[h-1,:] = post_u[h-5,:]
    # post_u[h-2,:] = post_u[h-4,:]
    # post_v[0,:] = post_v[4,:]
    # post_v[1,:] = post_v[3,:]
    # post_v[h-1,:] = post_v[h-5,:]
    # post_v[h-2,:] = post_v[h-4,:]

    # post_u[:,0] = post_u[:,4]
    # post_u[:,1] = post_u[:,3]
    # post_u[:,w-1] = post_u[:,w-5]
    # post_u[:,w-2] = post_u[:,w-4]
    # post_v[:,0] = post_v[:,4]
    # post_v[:,1] = post_v[:,3]
    # post_v[:,w-1] = post_v[:,w-5]
    # post_v[:,w-2] = post_v[:,w-4]

    # # 2nd round
    # u_xx = (post_u[2:h-2,1:w-3]+post_u[2:h-2,3:w-1]-2*post_u[2:h-2,2:w-2]) / pow(dx,2)
    # v_xy = (post_v[3:h-1,3:w-1]+post_v[1:h-3,1:w-3]-post_v[1:h-3,3:w-1]-post_v[3:h-1,1:w-3])/(4*dx*dy)
    # u_x4 = (post_u[2:h-2,4:w]-4*post_u[2:h-2,3:w-1]+6*post_u[2:h-2,2:w-2]-4*post_u[2:h-2,1:w-3]+post_u[2:h-2,0:w-4])/pow(dx,4)
    # v_x2y2 = ((post_v[3:h-1,3:w-1]+post_v[3:h-1,1:w-3]-2*post_v[3:h-1,2:w-2])\
    #         +(post_v[1:h-3,1:w-3]+post_v[1:h-3,3:w-1]-2*post_v[1:h-3,2:w-2])\
    #         -2*(post_v[2:h-2,1:w-3]+post_v[2:h-2,3:w-1]-2*post_v[2:h-2,2:w-2]))/(pow(dx,2)*pow(dy,2))
    # hgu = (u_xx+v_xy)-lamda_s*(u_x4+v_x2y2)-lamda_x*(post_u[2:h-2,2:w-2]-lateral_disp[2:h-2,2:w-2])

    # v_yy = (post_v[1:h-3,2:w-2]+post_v[3:h-1,2:w-2]-2*post_v[2:h-2,2:w-2])/pow(dy,2)
    # u_xy = (post_u[3:h-1,3:w-1]+post_u[1:h-3,1:w-3]-post_u[1:h-3,3:w-1]-post_u[3:h-1,1:w-3])/(4*dx*dy)
    # v_y4 = (post_v[4:h,2:w-2]-4*post_v[3:h-1,2:w-2]+6*post_v[2:h-2,2:w-2]-4*post_v[1:h-3,2:w-2]+post_v[0:h-4,2:w-2])/pow(dy,4)
    # u_x2y2 = ((post_u[3:h-1,3:w-1]+post_u[3:h-1,1:w-3]-2*post_u[3:h-1,2:w-2])\
    #         +(post_u[1:h-3,1:w-3]+post_u[1:h-3,3:w-1]-2*post_u[1:h-3,2:w-2])\
    #         -2*(post_u[2:h-2,1:w-3]+post_u[2:h-2,3:w-1]-2*post_u[2:h-2,2:w-2]))/(pow(dx,2)*pow(dy,2))
    # hgv = (u_xy+v_yy)-lamda_s*(u_x2y2+v_y4)-lamda_y*(post_v[2:h-2,2:w-2]-axial_disp[2:h-2,2:w-2])

    # post_u[2:h-2,2:w-2] = 3/4*pre_u[2:h-2,2:w-2]+1/4*post_u[2:h-2,2:w-2]+1/4*dt*hgu
    # post_v[2:h-2,2:w-2] = 3/4*pre_v[2:h-2,2:w-2]+1/4*post_v[2:h-2,2:w-2]+1/4*dt*hgv

    # post_u[0,:] = post_u[4,:]
    # post_u[1,:] = post_u[3,:]
    # post_u[h-1,:] = post_u[h-5,:]
    # post_u[h-2,:] = post_u[h-4,:]
    # post_v[0,:] = post_v[4,:]
    # post_v[1,:] = post_v[3,:]
    # post_v[h-1,:] = post_v[h-5,:]
    # post_v[h-2,:] = post_v[h-4,:]

    # post_u[:,0] = post_u[:,4]
    # post_u[:,1] = post_u[:,3]
    # post_u[:,w-1] = post_u[:,w-5]
    # post_u[:,w-2] = post_u[:,w-4]
    # post_v[:,0] = post_v[:,4]
    # post_v[:,1] = post_v[:,3]
    # post_v[:,w-1] = post_v[:,w-5]
    # post_v[:,w-2] = post_v[:,w-4]

    # # 3rd round
    # u_xx = (post_u[2:h-2,1:w-3]+post_u[2:h-2,3:w-1]-2*post_u[2:h-2,2:w-2]) / pow(dx,2)
    # v_xy = (post_v[3:h-1,3:w-1]+post_v[1:h-3,1:w-3]-post_v[1:h-3,3:w-1]-post_v[3:h-1,1:w-3])/(4*dx*dy)
    # u_x4 = (post_u[2:h-2,4:w]-4*post_u[2:h-2,3:w-1]+6*post_u[2:h-2,2:w-2]-4*post_u[2:h-2,1:w-3]+post_u[2:h-2,0:w-4])/pow(dx,4)
    # v_x2y2 = ((post_v[3:h-1,3:w-1]+post_v[3:h-1,1:w-3]-2*post_v[3:h-1,2:w-2])\
    #         +(post_v[1:h-3,1:w-3]+post_v[1:h-3,3:w-1]-2*post_v[1:h-3,2:w-2])\
    #         -2*(post_v[2:h-2,1:w-3]+post_v[2:h-2,3:w-1]-2*post_v[2:h-2,2:w-2]))/(pow(dx,2)*pow(dy,2))
    # hgu = (u_xx+v_xy)-lamda_s*(u_x4+v_x2y2)-lamda_x*(post_u[2:h-2,2:w-2]-lateral_disp[2:h-2,2:w-2])

    # v_yy = (post_v[1:h-3,2:w-2]+post_v[3:h-1,2:w-2]-2*post_v[2:h-2,2:w-2])/pow(dy,2)
    # u_xy = (post_u[3:h-1,3:w-1]+post_u[1:h-3,1:w-3]-post_u[1:h-3,3:w-1]-post_u[3:h-1,1:w-3])/(4*dx*dy)
    # v_y4 = (post_v[4:h,2:w-2]-4*post_v[3:h-1,2:w-2]+6*post_v[2:h-2,2:w-2]-4*post_v[1:h-3,2:w-2]+post_v[0:h-4,2:w-2])/pow(dy,4)
    # u_x2y2 = ((post_u[3:h-1,3:w-1]+post_u[3:h-1,1:w-3]-2*post_u[3:h-1,2:w-2])\
    #         +(post_u[1:h-3,1:w-3]+post_u[1:h-3,3:w-1]-2*post_u[1:h-3,2:w-2])\
    #         -2*(post_u[2:h-2,1:w-3]+post_u[2:h-2,3:w-1]-2*post_u[2:h-2,2:w-2]))/(pow(dx,2)*pow(dy,2))
    # hgv = (u_xy+v_yy)-lamda_s*(u_x2y2+v_y4)-lamda_y*(post_v[2:h-2,2:w-2]-axial_disp[2:h-2,2:w-2])

    # post_u[2:h-2,2:w-2] = 1/3*pre_u[2:h-2,2:w-2]+2/3*post_u[2:h-2,2:w-2]+2/3*dt*hgu
    # post_v[2:h-2,2:w-2] = 1/3*pre_v[2:h-2,2:w-2]+2/3*post_v[2:h-2,2:w-2]+2/3*dt*hgv

    # post_u[0,:] = post_u[4,:]
    # post_u[1,:] = post_u[3,:]
    # post_u[h-1,:] = post_u[h-5,:]
    # post_u[h-2,:] = post_u[h-4,:]
    # post_v[0,:] = post_v[4,:]
    # post_v[1,:] = post_v[3,:]
    # post_v[h-1,:] = post_v[h-5,:]
    # post_v[h-2,:] = post_v[h-4,:]

    # post_u[:,0] = post_u[:,4]
    # post_u[:,1] = post_u[:,3]
    # post_u[:,w-1] = post_u[:,w-5]
    # post_u[:,w-2] = post_u[:,w-4]
    # post_v[:,0] = post_v[:,4]
    # post_v[:,1] = post_v[:,3]
    # post_v[:,w-1] = post_v[:,w-5]
    # post_v[:,w-2] = post_v[:,w-4]

    # end

    # hgu_mean = torch.mean(torch.abs(post_u[2:h-2,2:w-2]-lateral_disp[2:h-2,2:w-2]))
    # hgv_mean = torch.mean(torch.abs(post_v[2:h-2,2:w-2]-axial_disp[2:h-2,2:w-2]))
    # eps = 0.001
    # diff_hgu = torch.abs(post_u[2:h-2,2:w-2]-lateral_disp[2:h-2,2:w-2])
    # diff_hgv = torch.abs(post_v[2:h-2,2:w-2]-axial_disp[2:h-2,2:w-2])

    # hgu_mean = torch.mean(torch.pow(torch.pow(diff_hgu,2)+eps,0.5))
    # hgv_mean = torch.mean(torch.pow(torch.pow(diff_hgv,2)+eps,0.5))

    # return hgu_mean,hgv_mean