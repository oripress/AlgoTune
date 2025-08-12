        # Use RK45 with maximum speed tolerances
        # Pushed to absolute limit while passing verification (rtol=1e-5, atol=1e-8)
        rtol = 9.9e-6
        atol = 9e-8
        
        # Aggressive step sizes for maximum speed
        first_step = 1.0
        max_step = time_span