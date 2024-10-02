import numpy as np
from pdb import set_trace as st
import matplotlib.pyplot as plt
import matplotlib
from model import FCN_poro_as_one
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from custom_loss import poro_loss
import scipy.io
import pandas as pd


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# CUDA support
print('Initializing Network...')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('run with GPU')
else:
    device = torch.device('cpu')
    print('run with CPU')

def PrepareData(data_path):

    data = scipy.io.loadmat(data_path)


    x_star = data['xx']
    y_star = data['yy']
    ip_star = data['ip']
    ip_x_star = data['ip_x']
    ip_xx_star = data['ip_xx']
    ip_y_star = data['ip_y']
    ip_yy_star = data['ip_yy']
    iux_star = data['iux']
    iux_x_star = data['iux_x']
    iux_xx_star = data['iux_xx']
    iux_xy_star = data['iux_xy']
    iux_yy_star = data['iux_yy']
    iuy_star = data['iuy']
    iuy_xx_star = data['iuy_xx']
    iuy_y_star = data['iuy_y']
    iuy_yx_star = data['iuy_yx']
    iuy_yy_star = data['iuy_yy']
    rp_star = data['rp']
    rp_x_star = data['rp_x']
    rp_xx_star = data['rp_xx']
    rp_y_star = data['rp_y']
    rp_yy_star = data['rp_yy']
    rux_star = data['rux']
    rux_x_star = data['rux_x']
    rux_xx_star = data['rux_xx']
    rux_xy_star = data['rux_xy']
    rux_yy_star = data['rux_yy']
    ruy_star = data['ruy']
    ruy_xx_star = data['ruy_xx']
    ruy_y_star = data['ruy_y']
    ruy_yx_star = data['ruy_yx']
    ruy_yy_star = data['ruy_yy']

    rF1x_star = data['rF1x']
    iF1x_star = data['iF1x']
    rF1y_star = data['rF1y']
    iF1y_star = data['iF1y']
    rF2_star = data['rF2']
    iF2_star = data['iF2']




    x_starr = torch.from_numpy(np.array(x_star.flatten())).type(torch.FloatTensor)
    y_starr = torch.from_numpy(np.array(y_star.flatten())).type(torch.FloatTensor)
    ip_starr = torch.from_numpy(np.array(ip_star.flatten())).type(torch.FloatTensor)
    ip_x_starr = torch.from_numpy(np.array(ip_x_star.flatten())).type(torch.FloatTensor)
    ip_xx_starr = torch.from_numpy(np.array(ip_xx_star.flatten())).type(torch.FloatTensor)
    ip_y_starr = torch.from_numpy(np.array(ip_y_star.flatten())).type(torch.FloatTensor)
    ip_yy_starr = torch.from_numpy(np.array(ip_yy_star.flatten())).type(torch.FloatTensor)
    iux_starr = torch.from_numpy(np.array(iux_star.flatten())).type(torch.FloatTensor)
    iux_x_starr = torch.from_numpy(np.array(iux_x_star.flatten())).type(torch.FloatTensor)
    iux_xx_starr = torch.from_numpy(np.array(iux_xx_star.flatten())).type(torch.FloatTensor)
    iux_xy_starr = torch.from_numpy(np.array(iux_xy_star.flatten())).type(torch.FloatTensor)
    iux_yy_starr = torch.from_numpy(np.array(iux_yy_star.flatten())).type(torch.FloatTensor)
    iuy_starr = torch.from_numpy(np.array(iuy_star.flatten())).type(torch.FloatTensor)
    iuy_xx_starr = torch.from_numpy(np.array(iuy_xx_star.flatten())).type(torch.FloatTensor)
    iuy_y_starr = torch.from_numpy(np.array(iuy_y_star.flatten())).type(torch.FloatTensor)
    iuy_yx_starr = torch.from_numpy(np.array(iuy_yx_star.flatten())).type(torch.FloatTensor)
    iuy_yy_starr = torch.from_numpy(np.array(iuy_yy_star.flatten())).type(torch.FloatTensor)
    rp_starr = torch.from_numpy(np.array(rp_star.flatten())).type(torch.FloatTensor)
    rp_x_starr = torch.from_numpy(np.array(rp_x_star.flatten())).type(torch.FloatTensor)
    rp_xx_starr = torch.from_numpy(np.array(rp_xx_star.flatten())).type(torch.FloatTensor)
    rp_y_starr = torch.from_numpy(np.array(rp_y_star.flatten())).type(torch.FloatTensor)
    rp_yy_starr = torch.from_numpy(np.array(rp_yy_star.flatten())).type(torch.FloatTensor)
    rux_starr = torch.from_numpy(np.array(rux_star.flatten())).type(torch.FloatTensor)
    rux_x_starr = torch.from_numpy(np.array(rux_x_star.flatten())).type(torch.FloatTensor)
    rux_xx_starr = torch.from_numpy(np.array(rux_xx_star.flatten())).type(torch.FloatTensor)
    rux_xy_starr = torch.from_numpy(np.array(rux_xy_star.flatten())).type(torch.FloatTensor)
    rux_yy_starr = torch.from_numpy(np.array(rux_yy_star.flatten())).type(torch.FloatTensor)
    ruy_starr = torch.from_numpy(np.array(ruy_star.flatten())).type(torch.FloatTensor)
    ruy_xx_starr = torch.from_numpy(np.array(ruy_xx_star.flatten())).type(torch.FloatTensor)
    ruy_y_starr = torch.from_numpy(np.array(ruy_y_star.flatten())).type(torch.FloatTensor)
    ruy_yx_starr = torch.from_numpy(np.array(ruy_yx_star.flatten())).type(torch.FloatTensor)
    ruy_yy_starr = torch.from_numpy(np.array(ruy_yy_star.flatten())).type(torch.FloatTensor)

    rF1x_starr = torch.from_numpy(np.array(rF1x_star.flatten())).type(torch.FloatTensor)
    iF1x_starr = torch.from_numpy(np.array(iF1x_star.flatten())).type(torch.FloatTensor)
    rF1y_starr = torch.from_numpy(np.array(rF1y_star.flatten())).type(torch.FloatTensor)
    iF1y_starr = torch.from_numpy(np.array(iF1y_star.flatten())).type(torch.FloatTensor)
    rF2_starr = torch.from_numpy(np.array(rF2_star.flatten())).type(torch.FloatTensor)
    iF2_starr = torch.from_numpy(np.array(iF2_star.flatten())).type(torch.FloatTensor)




    return x_starr, y_starr, ip_starr, ip_x_starr, ip_xx_starr, ip_y_starr, ip_yy_starr, iux_starr, iux_x_starr, iux_xx_starr, iux_xy_starr, iux_yy_starr, iuy_starr, iuy_xx_starr, iuy_y_starr, iuy_yx_starr, iuy_yy_starr, rp_starr, rp_x_starr, rp_xx_starr, rp_y_starr, rp_yy_starr, rux_starr, rux_x_starr, rux_xx_starr, rux_xy_starr, rux_yy_starr, ruy_starr, ruy_xx_starr, ruy_y_starr, ruy_yx_starr, ruy_yy_starr, rF1x_starr, iF1x_starr, rF1y_starr, iF1y_starr, rF2_starr, iF2_starr


def main():
    MODE = 'train' 



    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')


    print('Preparing dataset:')
    x_train1, y_train1, ip_train1, ip_x_train1, ip_xx_train1, ip_y_train1, ip_yy_train1, iux_train1, iux_x_train1, iux_xx_train1, iux_xy_train1, iux_yy_train1, iuy_train1, iuy_xx_train1, iuy_y_train1, iuy_yx_train1, iuy_yy_train1, rp_train1, rp_x_train1, rp_xx_train1, rp_y_train1, rp_yy_train1, rux_train1, rux_x_train1, rux_xx_train1, rux_xy_train1, rux_yy_train1, ruy_train1, ruy_xx_train1, ruy_y_train1, ruy_yx_train1, ruy_yy_train1, rF1x_train1, iF1x_train1, rF1y_train1, iF1y_train1, rF2_train1, iF2_train1 = PrepareData(data_path='data/poro_dataF_source1_normalized_based_on_rp.mat')


    EPOCH = 2000
    LEARNING_RATE_model = 1e-2
    LEARNING_RATE_weights = 1e-1




    input_size_NN =60*60*2
    alph = 0.12

    using_model = FCN_poro_as_one(input_size=input_size_NN).to(device)




    if MODE == 'train':


        print("Training Mode")


        Weightloss1 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        Weightloss2 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        Weightloss3 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        Weightloss4 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        Weightloss5 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        Weightloss6 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)



        ## put the weights for each loss in a array(matrix) to train the weights simutaneously with the NN model
        loss_parameters = [Weightloss1, Weightloss2, Weightloss3, Weightloss4, Weightloss5, Weightloss6]

        opt1 = optim.Adam(using_model.parameters(), lr=LEARNING_RATE_model, betas=(0.9,0.9999),eps=1e-6)
        opt2 = optim.Adam(loss_parameters, lr=LEARNING_RATE_weights, betas=(0.9,0.9999),eps=1e-6)


        L1loss = nn.L1Loss()
        mu_parameter_list = []
        lambda_parameter_list = []
        M_list = []
        phi_list = []
        kappa_list = []
        alpha_list = []

        e1_loss_list = []
        e2_loss_list = []
        e3_loss_list = []
        e4_loss_list = []
        e5_loss_list = []
        e6_loss_list = []


        l1_loss_list = []
        l2_loss_list = []
        l3_loss_list = []
        l4_loss_list = []
        l5_loss_list = []
        l6_loss_list = []


        loss_list = []


        adapt_weight1_list = []
        adapt_weight2_list = []
        adapt_weight3_list = []
        adapt_weight4_list = []
        adapt_weight5_list = []
        adapt_weight6_list = []





        plt.ion()
        for epoch in tqdm(range(1,EPOCH+1)):



            coef = 0
            input_ = torch.cat([x_train1, y_train1]).to(device)

            output1, output2, output3, output4, output5, output6 = using_model(input_)



            e1_loss, e2_loss, e3_loss, e4_loss, e5_loss, e6_loss = poro_loss(ip_train1.to(device),
                                        ip_x_train1.to(device),
                                        ip_xx_train1.to(device),
                                        ip_y_train1.to(device),
                                        ip_yy_train1.to(device),
                                        iux_train1.to(device),
                                        iux_x_train1.to(device),
                                        iux_xx_train1.to(device),
                                        iux_xy_train1.to(device),
                                        iux_yy_train1.to(device),
                                        iuy_train1.to(device),
                                        iuy_xx_train1.to(device),
                                        iuy_y_train1.to(device),
                                        iuy_yx_train1.to(device),
                                        iuy_yy_train1.to(device),
                                        rp_train1.to(device),
                                        rp_x_train1.to(device),
                                        rp_xx_train1.to(device),
                                        rp_y_train1.to(device),
                                        rp_yy_train1.to(device),
                                        rux_train1.to(device),
                                        rux_x_train1.to(device),
                                        rux_xx_train1.to(device),
                                        rux_xy_train1.to(device),
                                        rux_yy_train1.to(device),
                                        ruy_train1.to(device),
                                        ruy_xx_train1.to(device),
                                        ruy_y_train1.to(device),
                                        ruy_yx_train1.to(device),
                                        ruy_yy_train1.to(device),
                                        rF1x_train1.to(device),
                                        iF1x_train1.to(device),
                                        rF1y_train1.to(device),
                                        iF1y_train1.to(device),
                                        rF2_train1.to(device),
                                        iF2_train1.to(device),
                                        output1,
                                        output2,
                                        output3,
                                        output4,
                                        output5,
                                        output6,)
            

            l1 = loss_parameters[0].to(device)*e1_loss
            l2 = loss_parameters[1].to(device)*e2_loss
            l3 = loss_parameters[2].to(device)*e3_loss
            l4 = loss_parameters[3].to(device)*e4_loss
            l5 = loss_parameters[4].to(device)*e5_loss
            l6 = loss_parameters[5].to(device)*e6_loss

            loss = (l1+l2+l3+l4+l5+l6)/6





            if epoch == 1:
                l01 = l1.data
                l02 = l2.data
                l03 = l3.data
                l04 = l4.data
                l05 = l5.data
                l06 = l6.data


            opt1.zero_grad()
            loss.backward(retain_graph=True)



            param = list(using_model.parameters())

            G1R = torch.autograd.grad(l1, param[0], retain_graph=True, create_graph=True)
            G1 = torch.norm(G1R[0], 2)
            G2R = torch.autograd.grad(l2, param[0], retain_graph=True, create_graph=True)
            G2 = torch.norm(G2R[0], 2)
            G3R = torch.autograd.grad(l3, param[0], retain_graph=True, create_graph=True)
            G3 = torch.norm(G3R[0], 2)
            G4R = torch.autograd.grad(l4, param[0], retain_graph=True, create_graph=True)
            G4 = torch.norm(G4R[0], 2)
            G5R = torch.autograd.grad(l5, param[0], retain_graph=True, create_graph=True)
            G5 = torch.norm(G5R[0], 2)
            G6R = torch.autograd.grad(l6, param[0], retain_graph=True, create_graph=True)
            G6 = torch.norm(G6R[0], 2)


            G_avg= (G1+G2+G3+G4+G5+G6)/6

            # Calculating relative losses
            lhat1 = torch.div(l1,l01)
            lhat2 = torch.div(l2,l02)
            lhat3 = torch.div(l3,l03)
            lhat4 = torch.div(l4,l04)
            lhat5 = torch.div(l5,l05)
            lhat6 = torch.div(l6,l06)

            lhat_avg = (lhat1+lhat2+lhat3+lhat4+lhat5+lhat6)/6
            # Calculating relative inverse training rates for tasks
            inv_rate1 = torch.div(lhat1,lhat_avg)
            inv_rate2 = torch.div(lhat2,lhat_avg)
            inv_rate3 = torch.div(lhat3,lhat_avg)
            inv_rate4 = torch.div(lhat4,lhat_avg)
            inv_rate5 = torch.div(lhat5,lhat_avg)
            inv_rate6 = torch.div(lhat6,lhat_avg)


            C1 = G_avg*(inv_rate1)**alph
            C2 = G_avg*(inv_rate2)**alph
            C3 = G_avg*(inv_rate3)**alph
            C4 = G_avg*(inv_rate4)**alph
            C5 = G_avg*(inv_rate5)**alph
            C6 = G_avg*(inv_rate6)**alph



            C1 = C1.detach()
            C2 = C2.detach()
            C3 = C3.detach()
            C4 = C4.detach()
            C5 = C5.detach()
            C6 = C6.detach()


            opt2.zero_grad()
            Lgrad = L1loss(G1, C1)+L1loss(G2, C2)+L1loss(G3, C3)+L1loss(G4, C4)+L1loss(G5, C5)+L1loss(G6, C6)
            Lgrad.backward()

            # Updating loss weights
            opt2.step()

            # Updating the model weights
            opt1.step()

            # Renormalizing the losses weights
            coef = 6/(torch.abs(Weightloss1)+ torch.abs(Weightloss2)+ torch.abs(Weightloss3)+ torch.abs(Weightloss4)+ torch.abs(Weightloss5)+ torch.abs(Weightloss6))
            loss_parameters = [coef*torch.abs(Weightloss1), coef*torch.abs(Weightloss2), coef*torch.abs(Weightloss3), coef*torch.abs(Weightloss4), coef*torch.abs(Weightloss5), coef*torch.abs(Weightloss6)]




            print("Weights are:",loss_parameters[0], loss_parameters[1], loss_parameters[2], loss_parameters[3], loss_parameters[4], loss_parameters[5])
            print("loss_parameters are(coef*Weightloss_i):", loss_parameters)
            print(f'-------------Epoch {epoch}---------------')
            print(f'Loss:{loss.detach().cpu().numpy()}, e1 loss:{e1_loss.detach().cpu().numpy()}, e2 loss:{e2_loss.detach().cpu().numpy()}, e3 loss:{e3_loss.detach().cpu().numpy()}, e4 loss:{e4_loss.detach().cpu().numpy()}, e5 loss:{e5_loss.detach().cpu().numpy()}, e6 loss:{e6_loss.detach().cpu().numpy()}, mu_parameter:{output1.detach().cpu().numpy()}, lambda_parameter:{output2.detach().cpu().numpy()}, M:{output3.detach().cpu().numpy()}, phi:{output4.detach().cpu().numpy()}, kappa:{output5.detach().cpu().numpy()}, alpha:{output6.detach().cpu().numpy()} ')


            e1_loss_list.append(e1_loss.detach().cpu().numpy())
            e2_loss_list.append(e2_loss.detach().cpu().numpy())
            e3_loss_list.append(e3_loss.detach().cpu().numpy())
            e4_loss_list.append(e4_loss.detach().cpu().numpy())
            e5_loss_list.append(e5_loss.detach().cpu().numpy())
            e6_loss_list.append(e6_loss.detach().cpu().numpy())


            l1_loss_list.append(l1.detach().cpu().numpy())
            l2_loss_list.append(l2.detach().cpu().numpy())
            l3_loss_list.append(l3.detach().cpu().numpy())
            l4_loss_list.append(l4.detach().cpu().numpy())
            l5_loss_list.append(l5.detach().cpu().numpy())
            l6_loss_list.append(l6.detach().cpu().numpy())



            adapt_weight1_list.append(loss_parameters[0].detach().cpu().numpy())
            adapt_weight2_list.append(loss_parameters[1].detach().cpu().numpy())
            adapt_weight3_list.append(loss_parameters[2].detach().cpu().numpy())
            adapt_weight4_list.append(loss_parameters[3].detach().cpu().numpy())
            adapt_weight5_list.append(loss_parameters[4].detach().cpu().numpy())
            adapt_weight6_list.append(loss_parameters[5].detach().cpu().numpy())


            mu_parameter_list.append(output1.detach().cpu().numpy())
            lambda_parameter_list.append(output2.detach().cpu().numpy())
            M_list.append(output3.detach().cpu().numpy())
            phi_list.append(output4.detach().cpu().numpy())
            kappa_list.append(output5.detach().cpu().numpy())
            alpha_list.append(output6.detach().cpu().numpy())


            loss_list.append(loss.detach().cpu().numpy())



        plt.figure('Loss')
        plt.subplot(1,2,1)
        plt.title('Weighted Loss')
        plt.plot(np.log10(loss_list))
        plt.subplot(1,2,2)
        plt.title('Sub Losses')
        plt.plot(e1_loss_list,'r-.', label='e1 Loss')
        plt.plot(e2_loss_list,'b--',label='e2 Loss')
        plt.plot(e3_loss_list,'g--',label='e3 Loss')
        plt.plot(e4_loss_list,'c--',label='e4 Loss')
        plt.plot(e5_loss_list,'m--',label='e5 Loss')
        plt.plot(e6_loss_list,'y--',label='e6 Loss')


        plt.legend()
        plt.savefig('./logs/loss.png')
        plt.show()


        loss_list_prediction = pd.DataFrame(loss_list)
        loss_list_prediction.to_csv("./logs/loss_list_prediction.csv")

        e1_loss_list_prediction = pd.DataFrame(e1_loss_list)
        e1_loss_list_prediction.to_csv("./logs/e1_loss_list_prediction.csv")

        e2_loss_list_prediction = pd.DataFrame(e2_loss_list)
        e2_loss_list_prediction.to_csv("./logs/e2_loss_list_prediction.csv")

        e3_loss_list_prediction = pd.DataFrame(e3_loss_list)
        e3_loss_list_prediction.to_csv("./logs/e3_loss_list_prediction.csv")

        e4_loss_list_prediction = pd.DataFrame(e4_loss_list)
        e4_loss_list_prediction.to_csv("./logs/e4_loss_list_prediction.csv")

        e5_loss_list_prediction = pd.DataFrame(e5_loss_list)
        e5_loss_list_prediction.to_csv("./logs/e5_loss_list_prediction.csv")

        e6_loss_list_prediction = pd.DataFrame(e6_loss_list)
        e6_loss_list_prediction.to_csv("./logs/e6_loss_list_prediction.csv")


        plt.figure('Total Loss')
        plt.title('Total Loss')
        plt.plot(np.log10(loss_list))
        plt.savefig('./logs/total_loss.png')
        plt.show()

        plt.figure('e1 Loss')
        plt.title('e1 Loss')
        plt.plot(np.log10(e1_loss_list))
        plt.savefig('./logs/e1_loss_list.png')
        plt.show()

        plt.figure('e2 Loss')
        plt.title('e2 Loss')
        plt.plot(np.log10(e2_loss_list))
        plt.savefig('./logs/e2_loss_list.png')
        plt.show()

        plt.figure('e3 Loss')
        plt.title('e3 Loss')
        plt.plot(np.log10(e3_loss_list))
        plt.savefig('./logs/e3_loss_list.png')
        plt.show()

        plt.figure('e4 Loss')
        plt.title('e4 Loss')
        plt.plot(np.log10(e4_loss_list))
        plt.savefig('./logs/e4_loss_list.png')
        plt.show()

        plt.figure('e5 Loss')
        plt.title('e5 Loss')
        plt.plot(np.log10(e5_loss_list))
        plt.savefig('./logs/e5_loss_list.png')
        plt.show()

        plt.figure('e6 Loss')
        plt.title('e6 Loss')
        plt.plot(np.log10(e6_loss_list))
        plt.savefig('./logs/e6_loss_list.png')
        plt.show()


        plt.figure('l1 Loss')
        plt.title('l1 Loss')
        plt.plot(np.log10(l1_loss_list))
        plt.savefig('./logs/l1_loss_list.png')
        plt.show()

        plt.figure('l2 Loss')
        plt.title('l2 Loss')
        plt.plot(np.log10(l2_loss_list))
        plt.savefig('./logs/l2_loss_list.png')
        plt.show()

        plt.figure('l3 Loss')
        plt.title('l3 Loss')
        plt.plot(np.log10(l3_loss_list))
        plt.savefig('./logs/l3_loss_list.png')
        plt.show()

        plt.figure('l4 Loss')
        plt.title('l4 Loss')
        plt.plot(np.log10(l4_loss_list))
        plt.savefig('./logs/l4_loss_list.png')
        plt.show()

        plt.figure('l5 Loss')
        plt.title('l5 Loss')
        plt.plot(np.log10(l5_loss_list))
        plt.savefig('./logs/l5_loss_list.png')
        plt.show()

        plt.figure('l6 Loss')
        plt.title('l6 Loss')
        plt.plot(np.log10(l6_loss_list))
        plt.savefig('./logs/l6_loss_list.png')
        plt.show()



        plt.figure('mu_parameter_list')
        plt.title('mu_parameter_list')
        plt.plot(mu_parameter_list)
        plt.savefig('./logs/mu_parameter_list.png')
        plt.show()

        plt.figure('lambda_parameter_list')
        plt.title('lambda_parameter_list')
        plt.plot(lambda_parameter_list)
        plt.savefig('./logs/lambda_parameter_list.png')
        plt.show()

        plt.figure('M_list')
        plt.title('M_list')
        plt.plot(M_list)
        plt.savefig('./logs/M_list.png')
        plt.show()

        plt.figure('phi_list')
        plt.title('phi_list')
        plt.plot(phi_list)
        plt.savefig('./logs/phi_list.png')
        plt.show()

        plt.figure('kappa_list')
        plt.title('kappa_list')
        plt.plot(kappa_list)
        plt.savefig('./logs/kappa_list.png')
        plt.show()

        plt.figure('alpha_list')
        plt.title('alpha_list')
        plt.plot(alpha_list)
        plt.savefig('./logs/alpha_list.png')
        plt.show()




        plt.figure('adapt_weight1')
        plt.title('adapt_weight1')
        plt.plot(adapt_weight1_list)
        plt.savefig('./logs/adapt_weight1_list.png')
        plt.show()

        plt.figure('adapt_weight2')
        plt.title('adapt_weight2')
        plt.plot(adapt_weight2_list)
        plt.savefig('./logs/adapt_weight2_list.png')
        plt.show()

        plt.figure('adapt_weight3')
        plt.title('adapt_weight3')
        plt.plot(adapt_weight3_list)
        plt.savefig('./logs/adapt_weight3_list.png')
        plt.show()

        plt.figure('adapt_weight4')
        plt.title('adapt_weight4')
        plt.plot(adapt_weight4_list)
        plt.savefig('./logs/adapt_weight4_list.png')
        plt.show()

        plt.figure('adapt_weight5')
        plt.title('adapt_weight5')
        plt.plot(adapt_weight5_list)
        plt.savefig('./logs/adapt_weight5_list.png')
        plt.show()

        plt.figure('adapt_weight6')
        plt.title('adapt_weight6')
        plt.plot(adapt_weight6_list)
        plt.savefig('./logs/adapt_weight6_list.png')
        plt.show()


        plt.figure('adapt_weights')
        plt.title('adapt_weights')
        plt.plot(adapt_weight1_list,'b', label='w1')
        plt.plot(adapt_weight2_list,'g',label='w2')
        plt.plot(adapt_weight3_list,'r',label='w3')
        plt.plot(adapt_weight4_list,'c',label='w4')
        plt.plot(adapt_weight5_list,'k',label='w5')
        plt.plot(adapt_weight6_list,'y',label='w6')

        plt.legend(loc='center right')
        plt.savefig('./logs/adapt_weights.png')
        plt.show()


        plt.figure('adapt_weights_without_legend')
        plt.title('adapt_weights_without_legend')
        plt.plot(adapt_weight1_list,'b', label='w1')
        plt.plot(adapt_weight2_list,'g',label='w2')
        plt.plot(adapt_weight3_list,'r',label='w3')
        plt.plot(adapt_weight4_list,'c',label='w4')
        plt.plot(adapt_weight5_list,'k',label='w5')
        plt.plot(adapt_weight6_list,'y',label='w6')

        plt.savefig('./logs/adapt_weights_without_legend.png')
        plt.show()


        plt.figure('log10_adapt_weights')
        plt.title('log10_adapt_weights')
        plt.plot(np.log10(adapt_weight1_list),'b', label='w1')
        plt.plot(np.log10(adapt_weight2_list),'g',label='w2')
        plt.plot(np.log10(adapt_weight3_list),'r',label='w3')
        plt.plot(np.log10(adapt_weight4_list),'c',label='w4')
        plt.plot(np.log10(adapt_weight5_list),'k',label='w5')
        plt.plot(np.log10(adapt_weight6_list),'y',label='w6')

        plt.legend(loc='center right')
        plt.savefig('./logs/log10_adapt_weights.png')
        plt.show()



        plt.figure('log10_adapt_weights_without_legend')
        plt.title('log10_adapt_weights_without_legend')
        plt.plot(np.log10(adapt_weight1_list),'b', label='w1')
        plt.plot(np.log10(adapt_weight2_list),'g',label='w2')
        plt.plot(np.log10(adapt_weight3_list),'r',label='w3')
        plt.plot(np.log10(adapt_weight4_list),'c',label='w4')
        plt.plot(np.log10(adapt_weight5_list),'k',label='w5')
        plt.plot(np.log10(adapt_weight6_list),'y',label='w6')

        plt.savefig('./logs/log10_adapt_weights_without_legend.png')
        plt.show()






if __name__ == '__main__':
    main()
