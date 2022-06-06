#include <fstream>
#include <sstream>
#include <queue>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

std::vector<std::string> classNames;
const int inpWidth = 640;
const int inpHeight = 640;
const float confThreshold = 0.2;
const float nmsThreshold = 0.4;

const float scale = 0.00392;
const cv::Scalar mean = cv::Scalar();
const bool swapRB = true;

inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale, 
                    const cv::Scalar& mean, bool swapRB);

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out, cv::dnn::Net& net, int backend);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

template <typename T>
class QueueFPS : public std::queue<T>
{
public:
	void push(const T &entry)
	{
		std::lock_guard<std::mutex> lock(mutex);
		std::queue<T>::push(entry);
	}

	T get()
	{
		std::lock_guard<std::mutex> lock(mutex);
		T entry = this->front();
		this->pop();
		return entry;
	}

	void clear()
	{
		std::lock_guard<std::mutex> lock(mutex);
		while (!this->empty())
			this->pop();
	}

private:
	std::mutex mutex;
};

int main(int argc, char *argv[])
{
    // Open file with classes names.

    std::string file = "cfg/yolov5s.names";
    std::ifstream ifs(file.c_str());
    std::string line;
    while (std::getline(ifs, line))
        classNames.push_back(line);
 
    std::string modelPath = "cfg/yolov5s.onnx";

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    int backend = cv::dnn::DNN_BACKEND_CUDA;
    int target = cv::dnn::DNN_TARGET_CUDA_FP16;
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();


	bool process = true;

	// Frames capturing thread
	QueueFPS<cv::Mat> framesQueue;
	std::thread framesThread([&](){
		cv::VideoCapture cap(0);
        cv::Mat frame;
        while (process)
        {
            cap >> frame;
            if (!frame.empty())
                framesQueue.push(frame.clone());
            else
                break;
        }});

    // Frames processing thread
    QueueFPS<cv::Mat> processedFramesQueue;
    QueueFPS<std::vector<cv::Mat> > predictionsQueue;
    std::thread processingThread([&](){
        std::queue<cv::AsyncArray> futureOutputs;
        cv::Mat blob;
        while (process)
        {
            // Get a next frame
            cv::Mat frame;
            {
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    framesQueue.clear();  // Skip the rest of frames
                }
            }

            // Process the frame
            if (!frame.empty())
            {
                preprocess(frame, net, cv::Size(inpWidth, inpHeight), scale, mean, swapRB);
                processedFramesQueue.push(frame);

                std::vector<cv::Mat> outs;
                net.forward(outs, outNames);
                predictionsQueue.push(outs);           
            }
        }});

    // Create a window
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_AUTOSIZE);
    while (cv::waitKey(1) < 0)
    {
        if (predictionsQueue.empty())
            continue;
        
        std::vector<cv::Mat> outs = predictionsQueue.get();
        cv::Mat frame = processedFramesQueue.get();

        postprocess(frame, outs, net, backend);

        imshow(kWinName, frame);
    }

    framesThread.join();
    processingThread.join();
    process = false;
	return 0;
}

inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB)
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;

    cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false);

    // Run a model.
    net.setInput(blob, "", scale, mean);
}


void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend)
{
    // Network produces output blob with a shape 1x25200xN
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes; 

	CV_Assert(outs.size() > 0);
	float ratio_h = (float)frame.rows / inpHeight;
	float ratio_w = (float)frame.cols / inpWidth;

	for (size_t k = 0; k < outs.size(); k++)
	{
		float* data = (float*)outs[k].data;
		for (size_t i = 0; i < outs[k].total(); i += outs[k].step1(1))
		{
			float box_score = data[i + 4];
			cv::Mat scores(1, outs[k].step1(1) - 5, CV_32F, data + i + 5);
			cv::Point classIdPoint;
			double max_class_socre;
			cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			float confidence = box_score * max_class_socre;
			if (confidence > confThreshold)
			{
				float centerX = data[i];
				float centerY = data[i + 1];
				float width = data[i + 2];
				float height = data[i + 3];
				int left = (centerX - width / 2) * ratio_w;;
				int top = (centerY - height / 2) * ratio_h;
				width *= ratio_w;
				height *= ratio_h;
				classIds.push_back(classIdPoint.x);
				confidences.push_back(confidence);
				boxes.push_back(cv::Rect(left, top, (int)width, (int)height));
			}
		}
	}
 
     //NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
     //or NMS is required if number of outputs > 1
    if (backend != cv::dnn::DNN_BACKEND_OPENCV)
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        cv::Rect box = boxes[idx];
         drawPred(classIds[idx], confidences[idx], box.x, box.y,
                  box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    std::string label = cv::format("%.2f", conf);
    if (!classNames.empty())
    {
        CV_Assert(classId < (int)classNames.size());
        label = classNames[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = cv::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - labelSize.height),
        cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}