package com.tensorflow.android.classifier;

import android.Manifest;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.tensorflow.android.audio.features.WavFile;
import com.tensorflow.android.recorder.AudioRecorder;
import com.tensorflow.android.audio.features.MyMFCC;
import com.tensorflow.android.recorder.FileUtils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;



public class MainActivity extends AppCompatActivity {

    private AudioRecorder audioRecorder;
    private Button button_start;
    private TextView voice_text = null;
    private TextView recording_indictor = null;

    private SpeechUtils speechUtils ;

    private final static String filePath =  Environment.getExternalStorageDirectory().getAbsolutePath() + "/eth_demo/wav/eth_hw.wav";
    private final static String cutFilePath =  Environment.getExternalStorageDirectory().getAbsolutePath() + "/eth_demo/wav/eth_hw_2s.wav";

    private final static int audioBufferSize = 442100;
    private final static int bufferOverlap = 0;
    private final static int amountOfMelFilters = 0;
    private final static int nMfcc = 40;
    private final static float lowerFilterFreq = 0f;
    private final static float upperFilterFreq = 200000f;

    private int sampleRate = 22050;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.RECORD_AUDIO, Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS}, 5);

        speechUtils = new SpeechUtils(this);

        button_start = (Button) findViewById(R.id.RecordVoice);

        audioRecorder = AudioRecorder.getInstance();

        Toast.makeText(getApplicationContext(), "touch the button to record and untouch it upon finishing recording", Toast.LENGTH_LONG).show();
        voice_text = (TextView) findViewById(R.id.result_text);
        voice_text.setText("");
        recording_indictor = (TextView) findViewById(R.id.indicator);
        recording_indictor.setText("ready to record");

        button_start.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    // start recording.
                    audioRecorder.createDefaultAudio("eth_hw");
                    audioRecorder.startRecord(null);
                    recording_indictor.setText("staring to record");
                    return true;
                }
                if (event.getAction() == MotionEvent.ACTION_UP) {
                    // Stop recording and save file
                    audioRecorder.stopRecord();
                    if (!TextUtils.isEmpty(filePath)) {
                        // get the total length of recorded audio
                        File file = new File(filePath);
                        long duration = FileUtils.getWavLength(file);
                        Log.d("TAG", "duration is " + duration + " ms");

                        // if  less than 2000 ms return
                        if (duration < 100) {
                            recording_indictor.setText("please touch the button for a while");
                            return false;
                        }

                        if (duration < 2000) {
                            recording_indictor.setText("oops, it's too short, please try again");
                            return false;
                        }
                        recording_indictor.setText("finished recording");

                        // cut audio to get last 2 second and save it for furthure feature extraction.
                        FileUtils.cutAudio(filePath, cutFilePath, (int)duration-2000, (int)duration);

                        Log.d("TAG", "yeah, we have it recored, now predict the content ....");
                        String clsResult = ethVoiceClassification();
                        speechUtils.speakText("You sounds like " + clsResult);
                    } else {
                        Log.d("TAG", "sad, file not found");
                        Toast.makeText(getApplicationContext(), "recorded file not found", Toast.LENGTH_SHORT).show();
                    }
                    return true;
                }
                return false;
            }
        });
    }

    /* call audio dispatcher for mfcc calculation ,
    *   the problem currently we have is mfcc dimension is not 40 and
    *   can not pass into the network for predictionf
    * */
//
//     public void calculateMFCCThirdParty() {
//        AudioDispatcher audioDispatcher;
//        //new AndroidFFMPEGLocator(this);
//
//        try {
//            audioDispatcher = AudioDispatcherFactory.fromPipe(cutFilePath, sampleRate,
//                    audioBufferSize, bufferOverlap);
//        } catch (Exception e) {
//            e.printStackTrace();
//            return;
//        }
//
//        final MFCC mfccProcessor = new MFCC(audioBufferSize, sampleRate, nMfcc, amountOfMelFilters, lowerFilterFreq, upperFilterFreq);
//         audioDispatcher.addAudioProcessor(mfccProcessor);
//
//        final String[] voice_result = {"unknown"};
//        audioDispatcher.addAudioProcessor(new AudioProcessor() {
//
//            @Override // gets called on each audio frame
//            public boolean process(AudioEvent audioEvent) {
//                //final MyMFCC mfcc = new MyMFCC();
//
//                float[] mfccs = mfccProcessor.getMFCC();
//                System.out.print("mfcc : ");
//                int mfcc_size = mfccs.length;
//                System.out.println(mfcc_size);
//
//                // todo zaikun, add 2-d mfcc array to 2d cnn , for example 40x40 with padding to 0
//                int input_size = 3600;
//                float[] padded_mfcc = new float[input_size];
//
//                for (int i = 0; i < input_size; ++i) {
//                    if (i < mfcc_size ){
//                        System.out.println(i);
//                        padded_mfcc[i] = mfccs[i];
//                    }else {
//                        padded_mfcc[i] = 0f;
//                    }
//
//                }
//
//                voice_result[0] = predict(padded_mfcc);
//                return true;
//            }
//
//            @Override // gets called when end of the audio file was reached
//            public void processingFinished() {
//                //Toast.makeText( getApplicationContext(), "Predicted class : " + voice_result[0], Toast.LENGTH_LONG).show();
//                voice_text.setText("Predicted Voice type for last 2s : " + voice_result[0]);
//            }
//        });
//        new Thread(audioDispatcher, "audio dispacther").start();
//
//     }

    // use local mfcc algorithm to do the calculation
    public String calculateMFCCandPredict() {
         try {
             WavFile wavFile = WavFile.openWavFile(new File(cutFilePath));

             int mNumFrames = (int)wavFile.getNumFrames();
             int mSampleRate = (int)wavFile.getSampleRate();

             int mChannels = wavFile.getNumChannels();

             double[][] buffer = new double[mChannels][mNumFrames];
             //Array(mChannels) { DoubleArray(mNumFrames) }

             int frameOffset = 0;

             int loopCounter =  mNumFrames * mChannels / 4096 + 1;
             for (int i = 0; i < loopCounter; i++) {
                 frameOffset = wavFile.readFrames(buffer, mNumFrames, frameOffset);
             }

             //trimming the magnitude values to 5 decimal digits
             DecimalFormat df = new DecimalFormat("#.#####");

             df.setRoundingMode(RoundingMode.CEILING);
             double[] meanBuffer = new double[mNumFrames];

             for (int q = 0; q <  mNumFrames; q++) {
                 double frameVal = 0.0;
                 for (int p = 0; p < mChannels; p++) {
                     frameVal = frameVal + buffer[p][q];
                 }
                 meanBuffer[q] =  Double.parseDouble(df.format(frameVal / mChannels));
             }


             //MFCC java library.
             MyMFCC mfccConvert = new MyMFCC();
             mfccConvert.setSampleRate(mSampleRate);
             mfccConvert.setN_mfcc(nMfcc);
             float[] mfccInput = mfccConvert.process(meanBuffer);

             int input_size = 1280;
             float[] padded_mfcc = new float[input_size];

             Log.d("TAG", "mfcc : " + mfccInput.length);
             for (int i = 0; i < input_size; ++i) {
                 if (i < mfccInput.length ){
                     padded_mfcc[i] = mfccInput[i];
                 }else {
                     padded_mfcc[i] = 0f;
                 }

             }
            return predict(padded_mfcc);


         } catch (Exception e){

         }
         return "unknown";
    }

    public String ethVoiceClassification() {
        // filename
        try {

            //calculateMFCCThirdParty();
            String pred =  calculateMFCCandPredict();
            voice_text.setText("Predicted Voice type for last 2s : " + pred);
            return pred;

        } catch (Exception e) {
            Log.e("ethVoiceClassification", e.getMessage());
        }

        return "silence";
    }


    public String getPredictedValue(List<Recognition> predictedList){
        Recognition top1PredictedValue = predictedList.get(0);
        return top1PredictedValue.getTitle();
    }

    public String getModelPath() {
        return "model.tflite";
    }

    class RegComparator implements Comparator<Recognition> {
        public int compare(Recognition r1, Recognition r2) {
            if ( r1.getConfidence() < r2.getConfidence()){
                return 1;
            } else if (r1.getConfidence() > r2.getConfidence()) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    /** Gets the top-k results.  */
    public ArrayList<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        int MAX_RESULTS = 1;
        PriorityQueue<Recognition> pq = new PriorityQueue(
            MAX_RESULTS, new RegComparator());

        Iterator it = labelProb.entrySet().iterator();
        while(it.hasNext()){
            Map.Entry pair = (Map.Entry)(it.next());
            pq.add(new Recognition("" + pair.getKey(), (String)pair.getKey(), (float)pair.getValue()));
        }
        ArrayList<Recognition> recognitions = new ArrayList();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i <  recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }


    private MappedByteBuffer loadModelFile() throws IOException {
        String MODEL_ASSETS_PATH = getModelPath();
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd(MODEL_ASSETS_PATH) ;
        FileInputStream fileInputStream = new FileInputStream( assetFileDescriptor.getFileDescriptor() ) ;
        FileChannel fileChannel = fileInputStream.getChannel() ;
        long startoffset = assetFileDescriptor.getStartOffset() ;
        long declaredLength = assetFileDescriptor.getDeclaredLength() ;
        return fileChannel.map( FileChannel.MapMode.READ_ONLY , startoffset , declaredLength ) ;
    }

    public String predict(float[] meanMFCCValues) {

        String predictedResult = "unknown";
        try {
            Interpreter tflite = new Interpreter(loadModelFile());

            //for voice classification, input tensor should be of 1x40x1x1 shape , 40 is 40 mfcc features
            int imageTensorIndex = 0;
            int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
            DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

            int probabilityTensorIndex = 0;
            int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
            DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();


            //need to transform the MFCC 1d float buffer into 1x40x1x1 dimension tensor using TensorBuffer
            TensorBuffer inBuffer = TensorBuffer.createDynamic(imageDataType);
            inBuffer.loadArray(meanMFCCValues, imageShape);
            ByteBuffer inpBuffer = inBuffer.getBuffer();
            TensorBuffer outputTensorBuffer =
                    TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

            //inference
            tflite.run(inpBuffer, outputTensorBuffer.getBuffer());

            String ASSOCIATED_AXIS_LABELS = "labels.txt";
            List<String> associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);



            //Tensor processor for processing the probability values and to sort them based on the descending order of probabilities
            TensorProcessor probabilityProcessor;
            probabilityProcessor = new TensorProcessor.Builder().build(); //.add(new NormalizeOp(0, 1))
            if (null != associatedAxisLabels) {
                // Map of labels and their corresponding probability
                TensorLabel labels = new TensorLabel(associatedAxisLabels,
                        probabilityProcessor.process(outputTensorBuffer));

                // Create a map to access the result based on label
                Map<String, Float> floatMap = labels.getMapWithFloatValue();


                //function to retrieve the top K probability values, in this case 'k' value is 1.
                //retrieved values are storied in 'Recognition' object with label details.
                List<Recognition> resultPrediction = getTopKProbability(floatMap);

                //get the top 1 prediction from the retrieved list of top predictions
                predictedResult = getPredictedValue(resultPrediction);

            }
        } catch (Exception e) {
            Log.d("TAG", e.getMessage());
        }

        Log.d("TAG", "predicted : " + predictedResult);
        return predictedResult;
    }

}

//
//
//
//package com.tensorflow.android.classifier
//
//import android.os.Bundle
//import android.os.Environment
//import android.text.TextUtils
//import android.util.Log
//import android.view.View
//import android.widget.ArrayAdapter
//import android.widget.Spinner
//import android.widget.Toast
//import androidx.appcompat.app.AppCompatActivity
//import com.tensorflow.android.classifier.Recognition
//import com.tensorflow.android.audio.features.MFCC
//import com.tensorflow.android.audio.features.WavFile
//import com.tensorflow.android.audio.features.WavFileException
//import com.tensorflow.android.classifier.R
//import kotlinx.android.synthetic.main.activity_main.*
//import org.tensorflow.lite.DataType
//import org.tensorflow.lite.Interpreter
//import org.tensorflow.lite.support.common.FileUtil
//import org.tensorflow.lite.support.common.TensorProcessor
//import org.tensorflow.lite.support.common.ops.NormalizeOp
//import org.tensorflow.lite.support.label.TensorLabel
//import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
//import java.io.File
//import java.io.IOException
//import java.math.RoundingMode
//import java.nio.ByteBuffer
//import java.nio.MappedByteBuffer
//import java.text.DecimalFormat
//import java.util.*
//import kotlin.collections.ArrayList
//
//
//class MainActivity : AppCompatActivity() {
//
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//
//        //  val languages = resources.getStringArray(R.array.Languages)
//        val externalStorage: File = Environment.getExternalStorageDirectory()
//
//        val audioDirPath =  "/eth_demo/wav";
//
//
//        classify_button.setOnClickListener( View.OnClickListener {
//            var audioFilePath = Environment.getExternalStorageDirectory().getAbsolutePath() + audioDirPath + '/' + "eth_hw.wav";
//            if ( !TextUtils.isEmpty( audioFilePath ) ){
//
//                val result = classifyNoise(audioFilePath)
//                result_text.text = "Predicted Noise : $result"
//            }
//            else{
//                Toast.makeText( this@MainActivity, "Please enter a message.", Toast.LENGTH_LONG).show();
//            }
//        })
//
//    }
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
//            wavFile = WavFile.openWavFile(File(audioFilePath))
//            mNumFrames = wavFile.numFrames.toInt()
//            mSampleRate = wavFile.sampleRate.toInt()
//            mChannels = wavFile.numChannels
//            val buffer =
//                Array(mChannels) { DoubleArray(mNumFrames) }
//
//            var frameOffset = 0
//            val loopCounter: Int = mNumFrames * mChannels / 4096 + 1
//            for (i in 0 until loopCounter) {
//                frameOffset = wavFile.readFrames(buffer, mNumFrames, frameOffset)
//            }
//
//            //trimming the magnitude values to 5 decimal digits
//            val df = DecimalFormat("#.#####")
//            df.setRoundingMode(RoundingMode.CEILING)
//            val meanBuffer = DoubleArray(mNumFrames)
//            for (q in 0 until mNumFrames) {
//                var frameVal = 0.0
//                for (p in 0 until mChannels) {
//                    frameVal = frameVal + buffer[p][q]
//                }
//                meanBuffer[q] = df.format(frameVal / mChannels).toDouble()
//            }
//
//
//            //MFCC java library.
//            val mfccConvert = MFCC()
//            mfccConvert.setSampleRate(mSampleRate)
//            val nMFCC = 161 // todo set as private member
//            mfccConvert.setN_mfcc(nMFCC)
//            val mfccInput = mfccConvert.process(meanBuffer)
//            val nFFT = mfccInput.size / nMFCC
//            val mfccValues =
//                Array(nMFCC) { DoubleArray(nFFT) }
//
//            //loop to convert the mfcc values into multi-dimensional array
//            for (i in 0 until nFFT) {
//                var indexCounter = i * nMFCC
//                val rowIndexValue = i % nFFT
//                for (j in 0 until nMFCC) {
//                    mfccValues[j][rowIndexValue] = mfccInput[indexCounter].toDouble()
//                    indexCounter++
//                }
//            }
//
//            //code to take the mean of mfcc values across the rows such that
//            //[nMFCC x nFFT] matrix would be converted into
//            //[nMFCC x 1] dimension - which would act as an input to tflite model
//            meanMFCCValues = FloatArray(nMFCC)
//            Log.v("TAG" ,"the dimension of mfcc is  " +   meanMFCCValues.size);
//            for (p in 0 until nMFCC) {
//                var fftValAcrossRow = 0.0
//                for (q in 0 until nFFT) {
//                    fftValAcrossRow = fftValAcrossRow + mfccValues[p][q]
//                }
//                val fftMeanValAcrossRow = fftValAcrossRow / nFFT
//                meanMFCCValues[p] = fftMeanValAcrossRow.toFloat()
//            }
//
//
//        } catch (e: IOException) {
//            e.printStackTrace()
//        } catch (e: WavFileException) {
//            e.printStackTrace()
//        }
//
//        predictedResult = loadModelAndMakePredictions(meanMFCCValues)
//
//        return predictedResult
//
//    }
//
//
//    protected fun loadModelAndMakePredictions(meanMFCCValues : FloatArray) : String? {
//
//        var predictedResult: String? = "unknown"
//
//        //load the TFLite model in 'MappedByteBuffer' format using TF Interpreter
//        val tfliteModel: MappedByteBuffer =
//            FileUtil.loadMappedFile(this, getModelPath())
//        val tflite: Interpreter
//
//        /** Options for configuring the Interpreter.  */
//        val tfliteOptions =
//            Interpreter.Options()
//        tfliteOptions.setNumThreads(2)
//        tflite = Interpreter(tfliteModel, tfliteOptions)
//
//        //obtain the input and output tensor size required by the model
//        //for urban sound classification, input tensor should be of 1x40x1x1 shape
//        val imageTensorIndex = 0
//        val imageShape =
//            tflite.getInputTensor(imageTensorIndex).shape()
//        val imageDataType: DataType = tflite.getInputTensor(imageTensorIndex).dataType()
//        val probabilityTensorIndex = 0
//        val probabilityShape =
//            tflite.getOutputTensor(probabilityTensorIndex).shape()
//        val probabilityDataType: DataType =
//            tflite.getOutputTensor(probabilityTensorIndex).dataType()
//
//        //need to transform the MFCC 1d float buffer into 1x40x1x1 dimension tensor using TensorBuffer
//        val inBuffer: TensorBuffer = TensorBuffer.createDynamic(imageDataType)
//        inBuffer.loadArray(meanMFCCValues, imageShape)
//        val inpBuffer: ByteBuffer = inBuffer.getBuffer()
//        val outputTensorBuffer: TensorBuffer =
//            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
//        //run the predictions with input and output buffer tensors to get probability values across the labels
//        tflite.run(inpBuffer, outputTensorBuffer.getBuffer())
//
//
//        //Code to transform the probability predictions into label values
//        val ASSOCIATED_AXIS_LABELS = "labels.txt"
//        var associatedAxisLabels: List<String?>? = null
//        try {
//            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS)
//        } catch (e: IOException) {
//            Log.e("tfliteSupport", "Error reading label file", e)
//        }
//
//        //Tensor processor for processing the probability values and to sort them based on the descending order of probabilities
//        val probabilityProcessor: TensorProcessor = TensorProcessor.Builder()
//            .add(NormalizeOp(0.0f, 255.0f)).build()
//        if (null != associatedAxisLabels) {
//            // Map of labels and their corresponding probability
//            val labels = TensorLabel(
//                associatedAxisLabels,
//                probabilityProcessor.process(outputTensorBuffer)
//            )
//
//            // Create a map to access the result based on label
//            val floatMap: Map<String, Float> =
//                labels.getMapWithFloatValue()
//
//            //function to retrieve the top K probability values, in this case 'k' value is 1.
//            //retrieved values are storied in 'Recognition' object with label details.
//            val resultPrediction: List<Recognition>? = getTopKProbability(floatMap);
//
//            //get the top 1 prediction from the retrieved list of top predictions
//            predictedResult = getPredictedValue(resultPrediction)
//
//        }
//        return predictedResult
//
//    }
//
//
//    fun getPredictedValue(predictedList:List<Recognition>?): String?{
//        val top1PredictedValue : Recognition? = predictedList?.get(0)
//        return top1PredictedValue?.getTitle()
//    }
//
//    fun getModelPath(): String {
//        // you can download this file from
//        // see build.gradle for where to obtain this file. It should be auto
//        // downloaded into assets.
//        return "model.tflite"
//    }
//
//    /** Gets the top-k results.  */
//    protected fun getTopKProbability(labelProb: Map<String, Float>): List<Recognition>? {
//        // Find the best classifications.
//        val MAX_RESULTS: Int = 1
//        val pq: PriorityQueue<Recognition> = PriorityQueue(
//            MAX_RESULTS,
//            Comparator<Recognition> { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
//                java.lang.Float.compare(rhs.getConfidence(), lhs.getConfidence())
//            })
//        for (entry in labelProb.entries) {
//            pq.add(Recognition("" + entry.key, entry.key, entry.value))
//        }
//        val recognitions: ArrayList<Recognition> = ArrayList()
//        val recognitionsSize: Int = Math.min(pq.size, MAX_RESULTS)
//        for (i in 0 until recognitionsSize) {
//            recognitions.add(pq.poll())
//        }
//        return recognitions
//    }
//
//}