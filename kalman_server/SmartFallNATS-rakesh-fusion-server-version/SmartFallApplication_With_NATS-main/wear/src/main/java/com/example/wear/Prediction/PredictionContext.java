package com.example.wear.Prediction;

import android.content.Context;

import java.io.IOException;

import com.example.wear.config.SmartFallConfig;

/**
 * This class context class for top level prediction strategy. This class initializes the respective strategy class based on the
 * configuration provided in SmartFallConfig.java
 *
 * @author Bhargav Balusu (b_b515)
 * @version 1.0
 * @since 2022.05.15
 */
public class PredictionContext {

    /**
     * This method initializes the respective strategy configuration based on SmartFallConfig.MODEL_TYPE
     * @param context
     * @throws IOException
     */

    public static void initializeModel(Context context) throws IOException {
        switch(SmartFallConfig.MODEL_TYPE) {
            case  "PERSONALIZED" :
                PersonalizedStrategyContext.initialize(context);
                break;
            case "LSTM" :
                TensorFlowLiteLSTM.initialize(context);
                break;
            case "DEFAULT" :
                TensorFlowLiteEnsemble.initialize(context);
                break;
        }
    }

    /**
     * This method calls the respective makeInference method based on the strategy provided.
     * @param samples
     * @return calculated float value
     * @throws Exception
     */
    public static float makeInference(float[][] accInput, float[][] gyroInput) throws Exception{

        switch(SmartFallConfig.MODEL_TYPE) {
            case  "PERSONALIZED" :
                return PersonalizedStrategyContext.makeInference(accInput, gyroInput);
            default:
                throw new UnsupportedOperationException("Dual-sensor inference is only supported");
        }

    }

    /**
     * This method calls the respective getThreshold method based on the strategy provided.
     * @param samples
     * @return calculated float value
     * @throws Exception
     */
    public static float getThreshold(){

        switch(SmartFallConfig.MODEL_TYPE) {
            case  "PERSONALIZED" :
                return PersonalizedStrategyContext.getThreshold();
            case "LSTM" :
                return 0.19f;
            case "DEFAULT" :
                return 0.3f;
        }

        return 0;

    }
}
