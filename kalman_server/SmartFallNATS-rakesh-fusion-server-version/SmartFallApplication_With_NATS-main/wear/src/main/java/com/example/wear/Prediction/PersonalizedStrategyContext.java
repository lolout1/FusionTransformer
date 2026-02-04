package com.example.wear.Prediction;

import android.content.Context;

import java.io.IOException;

import com.example.wear.config.SmartFallConfig;

/**
 * This class is sub context class for personalized prediction strategy. This class initializes the respective strategy class based on the
 * configuration provided in SmartFallConfig.java
 *
 * @author Bhargav Balusu (b_b515)
 * @version 1.0
 * @since 2022.05.15
 */
public class PersonalizedStrategyContext {

    /**
     * This method initializes the respective strategy configuration based on SmartFallConfig.MODEL_TYPE
     * @param context
     * @throws IOException
     */
    public static void initialize(Context context) throws IOException {
        switch(SmartFallConfig.PERSONALIZED_STRATEGY) {
            case  "LSTM" :
                PersonalizedPredictionLSTM.initialize(context);
                break;
            case "ENSEMBLE" :
                PersonalizedPredictionEnsemble.initialize(context);
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

        switch(SmartFallConfig.PERSONALIZED_STRATEGY) {
            case  "LSTM" :
                return PersonalizedPredictionLSTM.makeInference(accInput, gyroInput);
            default:
                throw new UnsupportedOperationException("Dual-sensor inference is only supported for LSTM model.");

        }

    }

    public static float getThreshold(){

        switch(SmartFallConfig.PERSONALIZED_STRATEGY) {
            case "LSTM" :
                return PersonalizedPredictionLSTM.getThreshold();
            case "DEFAULT" :
                return PersonalizedPredictionEnsemble.getThreshold();
        }

        return 0;

    }
}
