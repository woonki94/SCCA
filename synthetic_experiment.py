import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA

import helper
from ALS_CCA import ALS_CCA
from SI_CCA import  SI_CCA
from SVD_CCA import SVD_CCA


def plot_convergence(suboptimality_gd, suboptimality_svrg, suboptimality_asvrg, suboptimality_si, suboptimality_si_asvrg):
    plt.figure(figsize=(6, 4))

    # Plot Gradient Descent (GD)
    plt.plot(suboptimality_gd, label="GD", color="blue", linestyle="-", linewidth=2)

    # Plot Stochastic Variance Reduced Gradient (SVRG)
    plt.plot(suboptimality_svrg, label="SVRG", color="red", linestyle="--", linewidth=2)

    # Plot Accelerated SVRG (ASVRG)
    plt.plot(suboptimality_asvrg, label="ASVRG", color="green", linestyle="-.", linewidth=2)

    plt.plot(suboptimality_si, label="SI-SVRG", color="orange", linestyle="-.", linewidth=2)

    plt.plot(suboptimality_si_asvrg, label="SI-ASVRG", color="grey", linestyle="-.", linewidth=2)



    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Passes")
    plt.ylabel("Suboptimality")
    plt.legend()
    plt.title("Convergence Plot")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    #TODO: need for more specific experiment.
    #Generate Data.
    X, Y = helper.generate_correlated_data(dx=10, dy=10, N=100, noise_level=2)# Simple struct Correlated data
    #Some points to be highlighted, as noise level goes up,  SVRG method gets faster
    #X,Y = helper.generate_strictly_low_rank_data(dx=10, dy=10, N=100,noise_level=2) #Low rank, Sparsity ensured Correlated data
    #X,Y = helper.generate_controlled_cca_data(dx=10, dy=10, N=100,noise_level=2) #Low rank, Sparsity ensured Correlated data

    #init algorithms
    cca = CCA(n_components=2,max_iter=10000)
    als_cca = ALS_CCA()
    als_cca_gd = ALS_CCA()
    als_cca_sgd = ALS_CCA()
    als_cca_asvrg = ALS_CCA()
    svd_cca = SVD_CCA()
    si_cca = SI_CCA()
    si_cca_asvrg = SI_CCA()

    #Find Direction
    cca.fit(X,Y)
    u_svd, v_svd, svd_elapsed_time = svd_cca.fit(X, Y)
    u_als, v_als, als_elapsed_time, suboptimality = als_cca.fit(X,Y)
    u_als_gd, v_als_gd, gd_elapsed_time, suboptimality_gd = als_cca_gd.fit(X, Y, method = "GD")
    u_als_sgd, v_als_sgd, sgd_elapsed_time,suboptimality_svrg = als_cca_sgd.fit(X, Y,method = "SVRG")
    u_als_asvrg, v_als_asvrg, asvrg_elapsed_time, suboptimality_asvrg = als_cca_asvrg.fit(X, Y, method="ASVRG")
    u_si, v_si, si_elapsed_time,suboptimality_si  = si_cca.fit(X,Y,method = "SVRG")
    u_si_asvrg, v_si_asvrg, si_elapsed_time_asvrg,suboptimality_si_asvrg  = si_cca_asvrg.fit(X,Y,method = "ASVRG")

    print(u_als.shape)
    print(u_si.shape)
    print(u_als_sgd)

    #plot_convergence(suboptimality_gd, suboptimality_svrg,suboptimality_asvrg,suboptimality_si,suboptimality_si_asvrg)
    #To see Equaivalent result, extract only the dominant cca component
    #from sklearn cca.
    X_proj, Y_proj = cca.transform(X,Y)
    X_proj = X_proj#[:, 0]  # First column
    Y_proj = Y_proj#[:, 0]  # First column

    X_proj_svd, Y_proj_svd =  svd_cca.transform(X,Y)
    X_proj_als, Y_proj_als =  als_cca.transform(X,Y)
    X_proj_gd, Y_proj_gd = als_cca_gd.transform(X, Y)
    X_proj_sgd, Y_proj_sgd =  als_cca_sgd.transform(X,Y)
    X_proj_asvrg, Y_proj_asvrg =  als_cca_asvrg.transform(X,Y)
    X_proj_si, Y_proj_si = si_cca.transform(X,Y)
    X_proj_si_asvrg, Y_proj_si_asvrg = si_cca_asvrg.transform(X,Y)


    print("===========time============")
    print("svd time: " ,svd_elapsed_time)
    print("als time: " ,als_elapsed_time)
    print("gd time: " , gd_elapsed_time)
    print("sgd time: " , sgd_elapsed_time)
    print("asvrg time: " , asvrg_elapsed_time)
    print("si_svrg time: " , si_elapsed_time)
    print("si_asvrg time: " , si_elapsed_time_asvrg)

    helper.plot_correlation_points(X_proj, Y_proj,title = "Built-in")
    corr = helper.canonical_correlation(X_proj, Y_proj)

    helper.plot_correlation_points(X_proj_svd,Y_proj_svd,title = "SVD")
    corr_svd = helper.canonical_correlation(X_proj_svd, Y_proj_svd)

    helper.plot_correlation_points(X_proj_als,Y_proj_als,title = "ALS")
    als_corr = helper.canonical_correlation(X_proj_als, Y_proj_als)

    helper.plot_correlation_points(X_proj_gd, Y_proj_gd,title = "ALS-GD")
    gd_corr = helper.canonical_correlation(X_proj_gd, Y_proj_gd)

    helper.plot_correlation_points(X_proj_sgd,Y_proj_sgd,title = "ALS-SVRG")
    sgd_corr = helper.canonical_correlation(X_proj_sgd,Y_proj_sgd)

    helper.plot_correlation_points(X_proj_asvrg, Y_proj_asvrg, title ="ALS-ASVRG")
    asvrg_corr = helper.canonical_correlation(X_proj_asvrg, Y_proj_asvrg)

    helper.plot_correlation_points(X_proj_si, Y_proj_si, title = "SI-SVRG")
    si_corr = helper.canonical_correlation(X_proj_si, Y_proj_si)

    helper.plot_correlation_points(X_proj_si_asvrg, Y_proj_si_asvrg, title = "SI-ASVRG")
    asvrg_si_corr = helper.canonical_correlation(X_proj_si_asvrg, Y_proj_si_asvrg)

    print("===========correlation============")
    print("Built-in SKLEARN Library Correlation: ", corr)
    print("SVD Correlation: ", corr_svd)
    print("ALS Correlation: " ,als_corr)
    print("GD ALS Correlation: ", gd_corr)
    print("SGD ALS Correlation: ", sgd_corr)
    print("ASVRG ALS Correlation: ", asvrg_corr)
    print("Shift and Invert Correlation: ", si_corr)
    print("Shift and Invert Correlation(ASVRG): ", asvrg_si_corr)


    pca = PCA(n_components=10)
    pca.fit(X.T)
    X_pca = pca.transform(X.T)

    print(X_pca.shape)