package com.tensorflow.android.recorder;

/**
 * Created by HXL on 16/8/11.
 * 获取录音的音频流,用于拓展的处理
 */
public interface RecordStreamListener {
    void recordOfByte(byte[] data, int begin, int end);
}
//
//
//    fun classifyNoise ( audioFilePath: String ): String? {
//
//        val mNumFrames: Int
//        val mSampleRate: Int
//        val mChannels: Int
//        var meanMFCCValues : FloatArray = FloatArray(1)
//
//        var predictedResult: String? = "Unknown"
//
//        var wavFile: WavFile? = null
//        try {
//        wavFile = WavFile.openWavFile(File(audioFilePath))
//        mNumFrames = wavFile.numFrames.toInt()
//        mSampleRate = wavFile.sampleRate.toInt()
//        mChannels = wavFile.numChannels
//        val buffer =
//        Array(mChannels) { DoubleArray(mNumFrames) }
//
//        var frameOffset = 0
//        val loopCounter: Int = mNumFrames * mChannels / 4096 + 1
//        for (i in 0 until loopCounter) {
//        frameOffset = wavFile.readFrames(buffer, mNumFrames, frameOffset)
//        }
//
//        //trimming the magnitude values to 5 decimal digits
//        val df = DecimalFormat("#.#####")
//        df.setRoundingMode(RoundingMode.CEILING)
//        val meanBuffer = DoubleArray(mNumFrames)
//        for (q in 0 until mNumFrames) {
//        var frameVal = 0.0
//        for (p in 0 until mChannels) {
//        frameVal = frameVal + buffer[p][q]
//        }
//        meanBuffer[q] = df.format(frameVal / mChannels).toDouble()
//        }
//
//
//        //MFCC java library.
//        val mfccConvert = MFCC()
//        mfccConvert.setSampleRate(mSampleRate)
//        val nMFCC = 40
//        mfccConvert.setN_mfcc(nMFCC)
//        val mfccInput = mfccConvert.process(meanBuffer)
//        val nFFT = mfccInput.size / nMFCC
//        val mfccValues =
//        Array(nMFCC) { DoubleArray(nFFT) }
//
//        //loop to convert the mfcc values into multi-dimensional array
//        for (i in 0 until nFFT) {
//        var indexCounter = i * nMFCC
//        val rowIndexValue = i % nFFT
//        for (j in 0 until nMFCC) {
//        mfccValues[j][rowIndexValue] = mfccInput[indexCounter].toDouble()
//        indexCounter++
//        }
//        }
//
//        //code to take the mean of mfcc values across the rows such that
//        //[nMFCC x nFFT] matrix would be converted into
//        //[nMFCC x 1] dimension - which would act as an input to tflite model
//        meanMFCCValues = FloatArray(nMFCC)
//        for (p in 0 until nMFCC) {
//        var fftValAcrossRow = 0.0
//        for (q in 0 until nFFT) {
//        fftValAcrossRow = fftValAcrossRow + mfccValues[p][q]
//        }
//        val fftMeanValAcrossRow = fftValAcrossRow / nFFT
//        meanMFCCValues[p] = fftMeanValAcrossRow.toFloat()
//        }
//
//
//        } catch (e: IOException) {
//        e.printStackTrace()
//        } catch (e: WavFileException) {
//        e.printStackTrace()
//        }
//
//        predictedResult = loadModelAndMakePredictions(meanMFCCValues)
//
//        return predictedResult
//
//        }
