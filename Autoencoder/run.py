import datetime
import pickle
import sys
import imports

""" ## choice controls the kind of simulation ##
choice = 2    : symbolswise channel autoencoder
choice = 3    : Autoencoder including pulseshaping 
choice = 1    : testing of functions
choice = 4    : serial training of radar functionalities
choice = 5    : joint training of comm and radar
choice = 6    : Radar tracking
choice = 7    : like 5, but with multiple targets
"""

begin_time = datetime.datetime.now()
#### Enable setting arguments from command line
if len(sys.argv)==1:
    choice = 18
elif len(sys.argv)==2:
    choice = int(sys.argv[1])
# elif len(sys.argv)==3:
#     choice=0
#     ## MIMO 
#     from training_routine_SNRsweep import *
#     logging.info("One simulation with SNR sweep and 1 targ, NN enc, 2 UEs")
#     M=torch.tensor([16], dtype=int).to(device)
#     wr = 0.7
#     beta = float(sys.argv[1])
#     num_ue = int(sys.argv[2])
#     #sigma_n=torch.tensor([0.1], dtype=float, device=device)
#     #sigma_c=100*sigma_n
#     #sigma_s=torch.tensor([0.1]).to(device)
#     SNR_s = torch.pow(10.0,torch.tensor([-10,0],device=device)/10)
#     SNR_c = torch.pow(10.0,torch.tensor([10,30],device=device)/10)
#     #training of exact beamform
#     logging.info("Modulation Symbols: "+str(M))
#     logging.info("SNR sensing = "+str(SNR_s))
#     logging.info("SNR Communication = "+str(SNR_c))
#     #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
#     enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,3]),weight_sens=wr,max_target=1,stage=1, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
#     enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,3]),weight_sens=wr,max_target=1,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
#     enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,10]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
#     enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
#     #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.0001,1,15]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
#     with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
#         pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==3:
    choice=0
    ## MIMO 
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN enc, 2 UEs")
    M=torch.tensor([16], dtype=int).to(device)
    wr = 0.2
    beta = float(sys.argv[1])
    num_ue = int(sys.argv[2])
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    # SNR_s = torch.pow(10.0,torch.tensor([0,0],device=device)/10)
    # SNR_c = torch.pow(10.0,torch.tensor([20,20],device=device)/10)
    SNR_s = torch.pow(10.0,torch.tensor([-10,0],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([15,20],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    logging.info("beta = "+str(beta))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    #enc_best,dec_best, beam_best, rad_rec_best = pickle.load( open( 'protosystem2.pkl', "rb" ) )

    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,30,0.001,1,5]),weight_sens=wr,max_target=1,stage=1, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.005,1,1]),weight_sens=wr,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,30,0.001,1,5]),weight_sens=wr,max_target=1,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.001,1,5]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)

    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.0001,1,15]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    beta = str(round(beta, 3)).translate(None, '.,')
    with open('/'+ beta +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==7:
    choice=0
    from training_routine_multitarget import *
    #print(sys.argv)
    M = torch.tensor([int(sys.argv[1])], dtype=int).to(device)
    sigma_n=torch.tensor([float(sys.argv[2])], dtype=float, device=device)
    sigma_c=torch.tensor([float(sys.argv[3])]).to(device)
    sigma_s=torch.tensor([float(sys.argv[4])]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    max_target = int(sys.argv[5])
    setbehaviour = sys.argv[6]
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling 1 parameter = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1]),weight_sens=0.9,max_target=max_target,stage=3, plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,K,train_params=cp.array([80,500,0.002]),weight_sens=1,max_target=max_target,NNs=[enc_best,dec_best, beam_best, rad_rec_best],stage=3, plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([80,50,0.001,1]),weight_sens=0.9,max_target=max_target,stage=None, plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1]),weight_sens=0.9,max_target=max_target,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour=setbehaviour, namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1]),weight_sens=0.9,max_target=max_target,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour=setbehaviour, namespace=namespace) 
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==8:
    choice=0
    from training_routine_multitarget import *
    #print(sys.argv)
    M = torch.tensor([int(sys.argv[1])], dtype=int).to(device)
    sigma_n=torch.tensor([float(sys.argv[2])], dtype=float, device=device)
    sigma_c=torch.tensor([float(sys.argv[3])]).to(device)
    sigma_s=torch.tensor([float(sys.argv[4])]).to(device)
    max_target = int(sys.argv[5])
    setbehaviour = sys.argv[6]
    beamloss = int(sys.argv[7])
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling 1 parameter = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([100,50,0.005,1]),weight_sens=0.9,max_target=max_target,stage=None, plotting=True,setbehaviour=setbehaviour, namespace=namespace, loss_beam=beamloss)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([50,50,0.01,1]),weight_sens=0.9,max_target=max_target,NNs=[enc_best,dec_best, beam_best, rad_rec_best],stage=None, plotting=True,setbehaviour=setbehaviour, namespace=namespace, loss_beam=beamloss)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==4:
    choice=0
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN or QAM enc")
    enctype = str(sys.argv[1])
    M=torch.tensor([int(sys.argv[2])], dtype=int).to(device)
    wr = float(sys.argv[3])
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([5,30],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=3, plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    #setting levels
    wrstr = str(wr).replace(".","")
    while wrstr[-1]=="0":
        wrstr = wrstr[0:len(wrstr)-1]
    for up in range(15):
        enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0,up+1,up+1]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    with open(figdir+'/'+enctype+'_'+ wrstr +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==5:
    choice=0
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN enc")
    enctype = str(sys.argv[1])
    M=torch.tensor([int(sys.argv[2])], dtype=int).to(device)
    wr = float(sys.argv[3])
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([5,30],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    if device!='cpu':
        if enctype=="QAM":
            enc, dec, beam, rad_rec = pickle.load( open( 'set/final_1508/QAM/QAM_07.pkl', "rb" ) )
        else:
            enc, dec, beam, rad_rec = pickle.load( open( 'set/final_1508/NN/NN_07.pkl', "rb" ) )
        enc.to(device)
        dec.to(device)
        beam.to(device)
        rad_rec.to(device)
    else:
        if enctype=="QAM":
            enc, dec, beam, rad_rec = CPU_Unpickler( open( 'set/final_1508/QAM/QAM_07.pkl', "rb")).load()
        else:
            enc, dec, beam, rad_rec = CPU_Unpickler( open( 'set/final_1508/NN/NN_07.pkl', "rb")).load()
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=3, plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([100,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc,dec, beam, rad_rec], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    wrstr = str(wr).replace(".","")
    while wrstr[-1]=="0":
        wrstr = wrstr[0:len(wrstr)-1]
    with open(figdir+'/'+enctype+'_temp_'+ wrstr +'.pkl', 'wb') as fh: # saving intermediate results
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
    #setting levels
    for up in range(15):
        enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0,up+1,up+1]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    with open(figdir+'/'+enctype+'_'+ wrstr +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
else:
    print(len(sys.argv))



if choice == 0:
    pass
elif choice == 2:
    from training_routine import *
    M=torch.tensor([4], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float)
    sigma_c=torch.sqrt(100*sigma_n**2)
    sigma_r=torch.sqrt(5*sigma_n**2)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, P_r, const = train_network(M,sigma_n,sigma_c,sigma_r,train_params=cp.array([80,300,0.005]),weight_sens=0.9,stage=None, plotting=True)


    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 4:
    from training_routine_multitarget import *
    M=torch.tensor([8], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float)
    sigma_c=10*sigma_n
    K=torch.tensor([200]).to(device)

    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,K,train_params=cp.array([20,30,0.01,3]),weight_sens=0.6,max_target=2,stage=3, plotting=True)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs2,gmi_exact2, P_r2, const = train_network(M,sigma_n,sigma_c,sigma_r,train_params=cp.array([40,100,0.01]),weight_sens=0.09,NNs=[enc_best,dec_best, beam_best, rad_rec_best],stage=2, plotting=True)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs3,gmi_exact3, P_r3, const = train_network(M,sigma_n,sigma_c,K,train_params=cp.array([20,30,0.01,3]),weight_sens=0.6,max_target=2,NNs=[enc_best,dec_best, beam_best, rad_rec_best],stage=1, plotting=True)
    #GMI_all = np.array([torch.sum(gmi_exact1,axis=1).detach().cpu().numpy(),torch.sum(gmi_exact2,axis=1).detach().cpu().numpy(),torch.sum(gmi_exact3,axis=1).detach().cpu().numpy()]).flatten()
    #P_d = np.array([P_r1[0].detach().cpu().numpy(),P_r2[0].detach().cpu().numpy(),P_r3[0].detach().cpu().numpy()]).flatten()
    #P_e = np.array([P_r1[1].detach().cpu().numpy(),P_r2[1].detach().cpu().numpy(),P_r3[1].detach().cpu().numpy()]).flatten()

    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 5:
    from training_routine import *
    M=torch.tensor([4], dtype=int)
    sigma_n=torch.tensor([5], dtype=float)
    sigma_c=100*sigma_n
    sigma_r=0.1*sigma_n
    #training of exact beamform
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_r,train_params=cp.array([60,300,0.008]),weight_sens=0.5,stage=None, plotting=True)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 7:
    from training_routine_multitarget import *
    logging.info("One simulation with 4 Targets and permute")
    M=torch.tensor([4], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    loss_beam = 0
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,np.sqrt(2)*sigma_s,train_params=cp.array([30,30,0.01,1]),weight_sens=1,max_target=1,stage=None, plotting=True,setbehaviour="none", namespace=namespace, loss_beam=loss_beam)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 8:
    from training_routine_multitarget import *
    t1 = torch.tensor([[1],[5],[-1],[4],[3]])
    t2 = torch.tensor([[1],[1],[5],[3],[4]])
    print(cdist(t1,t2))
elif choice == 9:
    ## Test of save_to_txt
    from training_routine_multitarget import *
    t =np.array([1,55,33,2,0.2])
    x = np.arange(5)
    x2 = np.array([0,0.5,0.5,0.5,0]) 
    l =["x","test","x2"]
    save_to_txt(np.array([x,t,x2]),2,l)
elif choice == 10:
    from training_routine_multitarget import *
    logging.info("One simulation with 1 Targets and no canc")
    M=torch.tensor([16], dtype=int).to(device)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,20,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,20,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,20,0.001,1,8]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,20,0.001,1,10]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,20,0.001,1,12]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 11:
    from training_routine_SNRsweep import *
    logging.info("One simulation with 4 Targets and permute")
    M=torch.tensor([16], dtype=int).to(device)
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.tensor([0.1,10],device=device)
    SNR_c = torch.tensor([10,100],device=device)
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None, plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,8]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,10]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,12]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 12:
    from training_routine_multitarget import *
    logging.info("One simulation with 1 Targets and no canc")
    M=torch.tensor([8], dtype=int)
    sigma_n=torch.tensor([0.01], dtype=float, device=device)
    sigma_c=100*sigma_n
    sigma_s=torch.tensor([10*sigma_n]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    if device!='cpu':
        enc, dec, beam, rad_rec = pickle.load( open( 'set/final_1508/NN/NN_07.pkl', "rb" ) )
        enc.to(device)
        dec.to(device)
        beam.to(device)
        rad_rec.to(device)
    else:
        enc, dec, beam, rad_rec = CPU_Unpickler( open( 'set/final_1508/NN/NN_07.pkl', "rb")).load()

    enc_best,dec_best, beam_best, rad_rec_best = CPU_Unpickler( open( "figures/set/sortphi.pkl", "rb")).load()
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([10,20,0.001,1]),weight_sens=0.9,max_target=3,stage=1, plotting=True,setbehaviour="setloss", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([10,20,0.001,1]),weight_sens=0.9,max_target=3,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="setloss", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([10,20,0.001,1,4]),weight_sens=0.9,max_target=3,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="setloss", namespace=namespace)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice==13:
    from training_routine_multitarget import *
    ## TEst of QAM encoder
    logging.info("Test of QAM encoder")
    M=torch.tensor([64],dtype=int)
    messages = torch.randint(0, int(M.detach().cpu().numpy()), (200,))
    t, code = QAM_encoder(messages, M, encoding=True)
    chelp = []
    plt.scatter(torch.real(t), torch.imag(t))
    for l in range(M):
        chelp.append("".join(str(code[l].detach().cpu().numpy())))
        t = QAM_encoder([l], M, encoding=False).detach().cpu().numpy()
        plt.annotate(chelp[l], (np.real(t), np.imag(t)))
    plt.show()
elif choice==14:
    from training_routine_multitarget import *
    logging.info("weighted cpr simulation with 2 Targets and permute")
    M=torch.tensor([16], dtype=int).to(device)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,1]),weight_sens=0.9,max_target=2,stage=None, plotting=True,setbehaviour="permute", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,3]),weight_sens=0.9,max_target=2,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="permute", namespace=namespace)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,5]),weight_sens=0.9,max_target=2,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="permute", namespace=namespace)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 15:
    from training_routine_multitarget import *
    logging.info("One simulation with 1 Targets, QAM enc, finetuning")
    M=torch.tensor([16], dtype=int).to(device)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    if device=='cuda':
        enc_best, dec_best, beam_best, rad_rec_best = pickle.load( open( "set/QAM_convinput2.pkl", "rb" ) )
    else:
        enc_best, dec_best, beam_best, rad_rec_best =  CPU_Unpickler( open( "set/QAM_convinput2.pkl", "rb")).load()
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None, NNs=[enc_best,dec_best, beam_best, rad_rec_best],plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,8]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,10]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,12]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 16:
    from training_routine_multitarget import *
    logging.info("One simulation with 1 Targets, NN enc, finetuning")
    M=torch.tensor([16], dtype=int).to(device)
    sigma_n=torch.tensor([0.1], dtype=float, device=device)
    sigma_c=10*sigma_n
    sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = sigma_s**2/sigma_n**2
    SNR_c = sigma_c**2/sigma_n**2
    if device=='cuda':
        enc_best, dec_best, beam_best, rad_rec_best = pickle.load( open( "set/NN_convinput2.pkl", "rb" ) )
    else:
        enc_best, dec_best, beam_best, rad_rec_best =  CPU_Unpickler( open( "set/NN_convinput2.pkl", "rb")).load()
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None, NNs=[enc_best,dec_best, beam_best, rad_rec_best],plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,8]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,10]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s, SNR_c,train_params=cp.array([80,40,0.001,1,12]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 17:
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, QAM enc")
    M=torch.tensor([16], dtype=int).to(device)
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-1,1],device=device))
    SNR_c = torch.pow(10.0,torch.tensor([0,2],device=device))
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.9,max_target=1,stage=3, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.9,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,5]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 18:
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN enc, 2 UEs")
    M=torch.tensor([16], dtype=int).to(device)
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([10,30],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0.005,1,1]),weight_sens=0.4,max_target=1,stage=1, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2,beta_corr=0.6)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0.005,1,1]),weight_sens=0.4,max_target=1,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2,beta_corr=0.6)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([10,10,0.001,1,1]),weight_sens=0.4,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2,beta_corr=0.6)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([10,10,0.001,1,5]),weight_sens=0.4,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2,beta_corr=0.6)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.0001,1,3]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.0001,1,15]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
else:
    pass
""" elif choice == 8:
    from training_routine_multitarget_conv import *
    logging.info("One simulation with 1 Targets and none setbehaviour")
    M=torch.tensor([8], dtype=int)
    sigma_n=torch.tensor([0.1], dtype=float).to(device)
    sigma_c=100*sigma_n
    sigma_s=torch.tensor([0.1]).to(device)
    #training of exact beamform
    logging.info("Detection and Localization of Multiple Targets")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("Additive Noise both channels: sigma_n = "+str(sigma_n))
    logging.info("Communication channel, fading param sigma_c = "+str(sigma_c))
    logging.info("Radar channel, fading param sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,sigma_n,sigma_c,sigma_s,train_params=cp.array([20,200,0.001,5]),weight_sens=0.8,max_target=1,stage=None, plotting=True,setbehaviour="none", namespace=namespace)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh) """
logging.info("Training duration is" + str(datetime.datetime.now()-begin_time))