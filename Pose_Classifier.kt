package com.bangkit.bangkitcapstone.pose.tflite

import android.content.res.AssetManager
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

class Pose_Classifier(assetManager: AssetManager, modelPath: String){
    private var interpreter: Interpreter

    // Init tflite interpreter
    init {
        val options = Interpreter.Options()
        options.setNumThreads(5)
        options.setUseNNAPI(true)
        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
        //lableList = loadLabelList(assetManager, labelPath)
    }

    // Load model from app/src/main/assets folder
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Predicts pose from input array and returns a float number
    fun predictPose(keypointsArray: FloatArray): Float {
        val input = keypointsArray
        val result = Array(1) {FloatArray(1)}
        interpreter.run(input, result)
        return result[0][0]
    }
}