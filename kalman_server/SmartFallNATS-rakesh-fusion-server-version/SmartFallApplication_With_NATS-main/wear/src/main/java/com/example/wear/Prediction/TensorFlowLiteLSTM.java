package com.example.wear.Prediction;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

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
 *  @author Bhargav Balusu (b_b515)
 *  @version 1.0
 *  @since 2022.05.15
 **/
public class TensorFlowLiteLSTM {

    /**
     * A string TAG for Debugging
     */
    private static final String TAG = "TensorFlowLiteLSTM";

    /**
     * A list of TensorFlow lite interpreters
     */
    private static ArrayList<Interpreter> interpreters = new ArrayList<>();

    /**
     * A list of model weights associated with each interpreter
     */
    private static float[] MODEL_WEIGHTS = {1.0f};

    /**
     * This method instantiates and appends a TensorFlow Lite interpreter using each model found
     * in the device assets folder
     *
     * @param context Context: App context
     */
    public static void initialize(Context context) throws IOException {
        interpreters.add(new Interpreter(loadModelFile(context,"saved_model-6.tflite")));
    }
    /**
     * This method locates and loads the model file from the assets folder
     *
     * @param context Context: App context
     * @param fileName String: The name of the model file located in the assets folder
     * @return ByteBuffer: The tensorflowlite model file
     */
    public static ByteBuffer loadModelFile(Context context, String fileName) throws IOException {

        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(fileName);
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        ByteBuffer byteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset,declaredLength);
        return byteBuffer;
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

            float[][][] flattenedSamples = flattenInputTo3D(samples);
            float[][] outputs = new float[1][1];
            outputs[0][0] = 0f;

            for (Interpreter interpreter : interpreters) {
                interpreter.run(flattenedSamples, outputs);   // make inference of beta samples
                inference += outputs[0][0] * MODEL_WEIGHTS[interpreters.indexOf(interpreter)];
            }

            return inference;
    }

    /**
     * This method is used to transforms samples from a two-dimensional float array
     * into a three-dimensional float array
     *
     * @param samples float: Two-dimensional float array
     * @return float: Three-dimensional float array
     */
    private static float[][][] flattenInputTo3D(float[][] samples) {
        float[][][] multiDimArray = new float[1][samples.length][samples[0].length];
        multiDimArray[0] = samples;
        return multiDimArray;
    }
}
