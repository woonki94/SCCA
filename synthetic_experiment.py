import helper
from ALS_CCA import ALS_CCA

# Example Usage
if __name__ == "__main__":
    #init algorithms
    als_cca = ALS_CCA()

    for i in range (10):
        X, Y = helper.generate_correlated_data(dx=100, dy=100, N=1000, noise_level=1)
        print(X.shape)
        print(Y.shape)

        u_als, v_als, elapsed_time = als_cca.fit(X, Y, SGD=False)
        u_als_sgd, v_als_sgd, sgd_elapsed_time = als_cca.fit(X, Y,SGD=True)

        #print("ALS CCA - u:", u_als.ravel())
        #print("ALS CCA - v:", v_als.ravel())
        #helper.plot_points(X,Y)
        #helper.plot_correlation(X, Y, u_als, v_als)
        helper.plot_correlation_points(X, Y, u_als, v_als)
        corr = helper.canonical_correlation(X,Y,u_als,v_als)

        helper.plot_correlation_points(X, Y, u_als_sgd, v_als_sgd)
        sgd_corr = helper.canonical_correlation(X, Y, u_als_sgd, v_als_sgd)

        print("Correlation: " ,corr)
        print("SGD ALS Correlation: ", sgd_corr)
