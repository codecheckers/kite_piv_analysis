def main():

    ##TODO: add an "if not exists" check to these processing scripts!
    # import transforming_paraview_output
    #
    # transforming_paraview_output.main()

    import calculating_noca_and_kutta

    #
    calculating_noca_and_kutta.main(n_points=2)

    # import convergence_study

    # convergence_study.main()

    ## now we can go over to plotting all figures

    ##TODO: also make sure to print the tables!


if __name__ == "__main__":
    main()
