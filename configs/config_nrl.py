config_NRL = {
    'adult': {
            'v0':{
                'encoder_hdims': None,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : None,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : None,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v1':{
                'encoder_hdims':[50],
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50],
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50],
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v2':{
                'encoder_hdims':[50]*2,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*2,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*2,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v3': {
                'encoder_hdims':[50]*3,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*3,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*3,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v4':{
                'encoder_hdims':[50],
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50],
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50],
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v5':{
                'encoder_hdims':[50]*2,
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*2,
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*2,
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v6': {
                'encoder_hdims':[50]*3,
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*3,
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*3,
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v7':{
                'encoder_hdims':[50],
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50],
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50],
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v8':{
                'encoder_hdims':[50]*2,
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*2,
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*2,
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            },
            'v9': {
                'encoder_hdims':[50]*3,
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*3,
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*3,
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 12
            }
            },
    'compas':{
            'v0':{
                'encoder_hdims': None,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : None,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : None,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v1':{
                'encoder_hdims':[50],
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50],
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50],
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v2':{
                'encoder_hdims':[50]*2,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*2,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*2,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v3': {
                'encoder_hdims':[50]*3,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*3,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*3,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v4':{
                'encoder_hdims':[50],
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50],
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50],
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v5':{
                'encoder_hdims':[50]*2,
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*2,
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*2,
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v6': {
                'encoder_hdims':[50]*3,
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*3,
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*3,
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v7':{
                'encoder_hdims':[50],
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50],
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50],
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v8':{
                'encoder_hdims':[50]*2,
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*2,
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*2,
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            },
            'v9': {
                'encoder_hdims':[50]*3,
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [50]*3,
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [50]*3,
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim':12
            }
            },
    'census_income_kdd':{
            
            'v0':{
                'encoder_hdims': None,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : None,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : None,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v1':{
                'encoder_hdims':[300],
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300],
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300],
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v2':{
                'encoder_hdims':[300]*2,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300]*2,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300]*2,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v3': {
                'encoder_hdims':[300]*3,
                'encoder_activation_hidden': None,
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300]*3,
                'decoder_activation_hidden': None,
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300]*3,
                'critic_activation_hidden': None,
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v4':{
                'encoder_hdims':[300],
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300],
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300],
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v5':{
                'encoder_hdims':[300]*2,
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300]*2,
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300]*2,
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v6': {
                'encoder_hdims':[300]*3,
                'encoder_activation_hidden': 'relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300]*3,
                'decoder_activation_hidden': 'relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300]*3,
                'critic_activation_hidden': 'relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v7':{
                'encoder_hdims':[300],
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300],
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300],
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v8':{
                'encoder_hdims':[399]*2,
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [399]*2,
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300]*2,
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            },
            'v9': {
                'encoder_hdims':[300]*3,
                'encoder_activation_hidden': 'leaky-relu',
                'encoder_activation_output': None,
                'encoder_p_dropout': None,
                'decoder_hdims' : [300]*3,
                'decoder_activation_hidden': 'leaky-relu',
                'decoder_activation_output': None, #since we need values between 0 and 1
                'decoder_p_dropout': None,
                'critic_hdims' : [300]*3,
                'critic_activation_hidden': 'leaky-relu',
                'critic_activation_output': None,
                'critic_p_dropout': None,
                'z_dim': 50
            }
            },
    # 'german': {
    #         'v_1':{
    #             'encoder_hdims':[38],
    #             'encoder_activation_hidden': None,
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'decoder_activation_hidden': None,
    #             'decoder_activation_output': 'sigmoid', #since we need values between 0 and 1
    #             'decoder_p_dropout': None,
    #             'decoder_hdims' : [38],
    #             'critic_hdims' : [38],
    #             'critic_activation_hidden': None,
    #             'critic_activation_output': 'sigmoid',
    #             'critic_p_dropout': None,
    #             'z_dim': 19
    #         },
    #         'v_2':{
    #             'encoder_hdims':[38]*2,
    #             'encoder_activation_hidden': 'relu',
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'decoder_hdims' : [38]*2,
    #             'decoder_activation_hidden': 'relu',
    #             'decoder_activation_output': 'sigmoid', #since we need values between 0 and 1
    #             'decoder_p_dropout': None,
    #             'critic_hdims' : [38]*2,
    #             'critic_activation_hidden': 'relu',
    #             'critic_activation_output': 'sigmoid',
    #             'critic_p_dropout': None,
    #             'z_dim': 19
    #         },
    #         'v_3': {
    #             'encoder_hdims':[38]*3,
    #             'encoder_activation_hidden': 'leaky-relu',
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'decoder_hdims' : [38]*3,
    #             'decoder_activation_hidden': 'leaky-relu',
    #             'decoder_activation_output': 'sigmoid', #since we need values between 0 and 1
    #             'decoder_p_dropout': None,
    #             'critic_hdims' : [38]*3,
    #             'critic_activation_hidden': 'leaky-relu',
    #             'critic_activation_output': 'sigmoid',
    #             'critic_p_dropout': None,
    #             'z_dim': 19
    #         }
    #         },
    # 'celeba':{
    #         'v_1':{
    #             'encoder_hdims':[254],
    #             'encoder_activation_hidden': None,
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'decoder_hdims' : [254],
    #             'decoder_activation_hidden': 'relu',
    #             'decoder_activation_output': 'sigmoid', #since we need values between 0 and 1
    #             'decoder_p_dropout': None,
    #             'critic_hdims' : [254],
    #             'critic_activation_hidden': 'leaky-relu',
    #             'critic_activation_output': 'sigmoid',
    #             'critic_p_dropout': None,
    #             'z_dim': 127
    #         },
    #         'v_2':{
    #             'encoder_hdims':[254],
    #             'encoder_activation_hidden': 'relu',
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'decoder_hdims' : [254],
    #             'decoder_activation_hidden': 'relu',
    #             'decoder_activation_output': 'sigmoid', #since we need values between 0 and 1
    #             'decoder_p_dropout': None,
    #             'critic_hdims' : [254]*2,
    #             'critic_activation_hidden': 'relu',
    #             'critic_activation_output': 'sigmoid',
    #             'critic_p_dropout': None,
    #             'z_dim': 127
    #         },
    #         'v_3': {
    #             'encoder_hdims':[254],
    #             'encoder_activation_hidden': 'leaky-relu',
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'encoder_activation_hidden': 'leaky-relu',
    #             'encoder_activation_output': None,
    #             'encoder_p_dropout': None,
    #             'decoder_hdims' : [254],
    #             'decoder_activation_hidden': 'leaky-relu',
    #             'decoder_activation_output': 'sigmoid', #since we need values between 0 and 1
    #             'decoder_p_dropout': None,
    #             'critic_hdims' : [254]*3,
    #             'critic_activation_hidden': 'leaky-relu',
    #             'critic_activation_output': 'sigmoid',
    #             'critic_p_dropout': None,
    #             'z_dim': 127
    #         }
    #         },
}
