package com.example.wear.Prediction;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

/**
 *  This class utilizes the TensorFlow-lite API to make inferences on
 *  float data.
 *
 *  @author Colin Campbell (c_c953)
 *  @version 1.0
 *  @since 2021.12.13
 **/
public class TensorFlowLiteEnsemble {

    /**
     * A string TAG for Debugging
     */
    private static final String TAG = "TensorFlowLiteEnsemble";

    /**
     * A list of TensorFlow lite interpreters
     */
    private static ArrayList<Interpreter> interpreters = new ArrayList<>();

    /**
     * A list of model weights associated with each interpreter
     */
    private static float[] MODEL_WEIGHTS = {1.3464559f, -0.320784f, -1.4609069f, 1.6781534f};


    /**
     * This method instantiates and appends a TensorFlow Lite interpreter using each model found
     * in the device assets folder
     *
     * @param context Context: App context
     */
    public static void initialize(Context context) throws IOException {
        interpreters.add(new Interpreter(loadModelFile(context,"10generic_adlone.tflite")));
        interpreters.add(new Interpreter(loadModelFile(context,"10generic_adltwo.tflite")));
        interpreters.add(new Interpreter(loadModelFile(context,"10generic_adlthree.tflite")));
        interpreters.add(new Interpreter(loadModelFile(context,"10generic_adlfour.tflite")));
    }

    /**
     * This method locates and loads the model file from the assets folder
     *
     * @param context Context: App context
     * @param fileName String: The name of the model file located in the assets folder
     * @return ByteBuffer: The tensorflowlite model file
     */
    private static ByteBuffer loadModelFile(Context context, String fileName) throws IOException {

        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(fileName);
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        ByteBuffer byteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset,declaredLength);
        return byteBuffer;
    }

    /**
     * This method is used to transforms samples from a two-dimensional float array
     * into a one-dimensional float array
     *
     * @param samples float: Two-dimensional float array
     * @return float: One-dimensional float array
     */
    public static float[] flattenInputs(float[][] samples) {

        float[] flattenedSamples = new float[samples.length * samples[0].length];
        for (int i = 0; i < flattenedSamples.length; i += samples[0].length) {
            for (int x = 0; x < samples[0].length; x++) {
                try {
                    flattenedSamples[i + x] = samples[ i / samples[0].length][x];
                }catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }
        }
        return flattenedSamples;
    }

    /**
     * This method performs a weighted sum inference on received float samples using
     * the list of interpreters.
     *
     * @param samples float: Two-dimensional float array of samples
     * @return float: The weighted sum inference made my the interpreters
     */
    public static float makeInference(float[][] samples) throws Exception {

        float inference = 0.0f;

        float[] flattenedSamples = flattenInputs(samples);
        float[][] outputs = new float[1][2];
        outputs[0][0] = 0f;
        outputs[0][1] = 0f;

        for (Interpreter interpreter : interpreters) {
            interpreter.run(flattenedSamples, outputs);   // make inference of beta samples
            inference += outputs[0][0] * MODEL_WEIGHTS[interpreters.indexOf(interpreter)];
        }

        return inference;
    }

}
