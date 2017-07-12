package application;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.text.Text;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the image segmentation process.
 * 
 * 
 */
public class ObjRecognitionController
{
	// FXML camera button
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// the FXML area for showing the mask
	@FXML
	private ImageView maskImage;
	// FXML slider for setting HSV ranges
	@FXML
	private Slider hueStart;
	@FXML
	private Slider hueStop;
	@FXML
	private Slider saturationStart;
	@FXML
	private Slider saturationStop;
	@FXML
	private Slider valueStart;
	@FXML
	private Slider valueStop;
	// FXML label to show the current values set with the sliders
	@FXML
	private Label hsvCurrentValues;
	@FXML
	private Text count;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive;
	
	// property for object binding
	private ObjectProperty<String> hsvValuesProp;
		
	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	private void startCamera()
	{
		// bind a text property with the string containing the current range of
		// HSV values for object detection
		hsvValuesProp = new SimpleObjectProperty<>();
		this.hsvCurrentValues.textProperty().bind(hsvValuesProp);
				
		// set a fixed width for all the image to show and preserve image ratio
		this.imageViewProperties(this.originalFrame, 400);
		this.imageViewProperties(this.maskImage, 400);
		
		if (!this.cameraActive)
		{
			// start the video capture
			this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame()
	{
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// init
					Mat blurredImage = new Mat();
					Mat hsvImage = new Mat();
					Mat mask = new Mat();
					
					// remove some noise
					Imgproc.blur(frame, blurredImage, new Size(7, 7));
					
					// convert the frame to HSV
					Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HSV);
					
					// get thresholding values from the UI
					// remember: H ranges 0-180, S and V range 0-255
					Scalar minValues = new Scalar(this.hueStart.getValue(), this.saturationStart.getValue(),
							this.valueStart.getValue());
					Scalar maxValues = new Scalar(this.hueStop.getValue(), this.saturationStop.getValue(),
							this.valueStop.getValue());
					
					// show the current selected HSV range
					String valuesToPrint = "Hue range: " + minValues.val[0] + "-" + maxValues.val[0]
							+ "\tSaturation range: " + minValues.val[1] + "-" + maxValues.val[1] + "\tValue range: "
							+ minValues.val[2] + "-" + maxValues.val[2];
					Utils.onFXThread(this.hsvValuesProp, valuesToPrint);
					
					// threshold HSV image
					Core.inRange(hsvImage, minValues, maxValues, mask);
					
					// show the partial output
					this.updateImageView(this.maskImage, Utils.mat2Image(mask));
					
					// find the contours and show them
					frame = this.generateContours(mask, frame);

				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.print("Exception during the image elaboration...");
				e.printStackTrace();
			}
		}
		
		return frame;
	}
	
	//helper method to find biggest contour
	private int findBiggestContour(List<MatOfPoint> contours)
	{
		int indexOfBiggestContour = -1;
		double sizeOfBiggestContour = 0;
		for (int i = 0; i < contours.size(); i++){
	        if(Imgproc.contourArea(contours.get(i)) > sizeOfBiggestContour){
	            sizeOfBiggestContour = Imgproc.contourArea(contours.get(i));
	            indexOfBiggestContour = i;
	        }
	    }
		return indexOfBiggestContour;
	}
	
	
	
	/**
	 * Given a binary image containing one or more closed surfaces, use it as a
	 * mask to find and highlight the objects contours
	 * 
	 * @param maskedImage
	 *            the binary image to be used as a mask
	 * @param frame
	 *            the original {@link Mat} image to be used for drawing the
	 *            objects contours
	 * @return the {@link Mat} image with the objects contours framed
	 */
	private Mat generateContours(Mat maskedImage, Mat frame)
	{
//		// init
//		List<MatOfPoint> contours = new ArrayList<>();
//		Mat hierarchy = new Mat();
//		
//		// find contours
//		Imgproc.findContours(maskedImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
//		
//		// if any contours exist...
//		if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
//		{
//			// for each contour, display it in blue
//			for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
//			{
//				Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0));
//			}
//		}
//		
//		return frame;

		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();

		
		Imgproc.findContours(maskedImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0,0));
		List<MatOfInt> hull = new ArrayList<MatOfInt>();

		List<MatOfInt4> defects = new ArrayList<MatOfInt4>();
        for(int i=0; i < contours.size(); i++){
            hull.add(new MatOfInt());
            defects.add(new MatOfInt4());
        }

        //generate the convex hull and defects 
        for(int i=0; i < contours.size(); i++){
            Imgproc.convexHull(contours.get(i), hull.get(i));
            Imgproc.convexityDefects(contours.get(i),hull.get(i), defects.get(i));
        }

        

        // Loop over all contours
        List<Point[]> hullpoints = new ArrayList<Point[]>();
        for(int i=0; i < hull.size(); i++){
            Point[] points = new Point[hull.get(i).rows()];

            // Loop over all points that need to be hulled in current contour
            for(int j=0; j < hull.get(i).rows(); j++){
                int index = (int)hull.get(i).get(j, 0)[0];
                points[j] = new Point(contours.get(i).get(index, 0)[0], contours.get(i).get(index, 0)[1]);
            }

            hullpoints.add(points);
        }

        // Convert Point arrays into MatOfPoint
        List<MatOfPoint> hullmop = new ArrayList<MatOfPoint>();
        for(int i=0; i < hullpoints.size(); i++){
            MatOfPoint mop = new MatOfPoint();
            mop.fromArray(hullpoints.get(i));
            hullmop.add(mop);
        }

        // Draw contours + hull results
        int biggestContourIndex = findBiggestContour(contours);
        int fingerCount = 1;
        Scalar color = new Scalar(0, 255, 0);   // Green
        for(int i=0; i < contours.size(); i++){

        	//choose only the biggest contour
        	if(i == biggestContourIndex){
        		Imgproc.drawContours(frame, contours, i, new Scalar(0,0,255),2);
        		for(int j=0; j< defects.get(i).toList().size()-3; j+=4)
        		{
//        			//store the depth of the defect
        			float depth = defects.get(i).toList().get(j+3) / 256;
        			if(depth > 10)
        			{
        				//store indexes of start, end, and far points
        				int startid = defects.get(i).toList().get(j);
        				//store the point on the contour as a Point object
        				Point startPt = contours.get(i).toList().get(startid);
        				int endid = defects.get(i).toList().get(j+1);
        				Point endPt = contours	.get(i).toList().get(endid);
        				int farid = defects.get(i).toList().get(j+2);
        				Point farPt = contours.get(i).toList().get(farid);
        				
        				if (isFinger(defects.get(i),contours.get(i),j)) {
//
        					if (fingerCount < 5)
        						fingerCount++;
        					System.out.println("Distance from start to far: " + distanceFormula(startPt,farPt));
        					System.out.println("Distance from end to far:   " + distanceFormula(endPt,farPt));
        					System.out.println("Angle of defect:            " + getAngle(startPt,endPt,farPt));
            				//draw line from start to end
            				Imgproc.line(frame,startPt,endPt,new Scalar(255,255,255),2);	
            				//draw line from start to far point
            				Imgproc.line(frame,startPt,farPt,new Scalar(255,255,255),2);	
            				//draw line from end to far point
            				Imgproc.line(frame,endPt,farPt,new Scalar(255,255,255),2);
            				//draw circle around far point
            				Imgproc.circle(frame,farPt,4,new Scalar(255,255,255),2);
        				}
        				else
        				{
	        				//draw line from start to end
	        				Imgproc.line(frame,startPt,endPt,new Scalar(255,0,0),2);	
	        				//draw line from start to far point
	        				Imgproc.line(frame,startPt,farPt,new Scalar(255,0,0),2);	
	        				//draw line from end to far point
	        				Imgproc.line(frame,endPt,farPt,new Scalar(255,0,0),2);
	        				//draw circle around far point
	        				Imgproc.circle(frame,farPt,4,new Scalar(255,0,0),2);
	        				
        				}
        			}
        		}
        		//draw convex hull of biggest contour
                Imgproc.drawContours(frame, hullmop, i, new Scalar(0,255,255),2);
                
        	}
        	else //draw smaller contours in green
        	{
        		Imgproc.drawContours(frame, contours, i, color);
        		
        	}
        }
        count.setText(fingerCount + " finger(s) detected");

        return frame;
        
        
	}
		
	//Pre: MatOfInt4 of defect list, MatOfPoint of hand contour, and index j of defect of interest
	private boolean isFinger(MatOfInt4 defect,MatOfPoint contour,int j)
	{
		Rect boundingRect= Imgproc.boundingRect(contour);
		int tolerance = boundingRect.height / 5;
		double angleTol = 95;	
		//store indexes of start, end, and far points
		int startid = defect.toList().get(j);
		//store the point on the contour as a Point object
		Point startPt = contour.toList().get(startid);
		int endid = defect.toList().get(j+1);
		Point endPt = contour.toList().get(endid);
		int farid = defect.toList().get(j+2);
		Point farPt = contour.toList().get(farid);
		
		if (distanceFormula(startPt,farPt)>tolerance && 
			distanceFormula(endPt,farPt)>tolerance && 
			getAngle(startPt,endPt,farPt) < angleTol &&
			endPt.y <= (boundingRect.y + boundingRect.height - boundingRect.height/4) &&
			startPt.y <= (boundingRect.y + boundingRect.height - boundingRect.height/4))
				return true;
		
		return false;
	}	
	
	
	
	//use Law of Cosines to find angle between 3 points
	private double getAngle(Point start, Point end, Point far)
	{
		//distance between start and far
		double a = distanceFormula(start,far);
		//distance between end and far
		double b = distanceFormula(end,far);
		//distance between start and end (side c of triangle)
		double c = distanceFormula(start,end);
		//Law of Cosines
		double angle = Math.acos((a*a + b*b - c*c) / (2*a*b));
		angle = angle*180/Math.PI;
		return angle;
	}
	
	private double distanceFormula(Point start, Point end)
	{
		return Math.sqrt(Math.abs(Math.pow(start.x-end.x, 2) + Math.pow(start.y-end.y, 2)));
	}
	
	/**
	 * Set typical {@link ImageView} properties: a fixed width and the
	 * information to preserve the original image ration
	 * 
	 * @param image
	 *            the {@link ImageView} to use
	 * @param dimension
	 *            the width of the image to set
	 */
	private void imageViewProperties(ImageView image, int dimension)
	{
		// set a fixed width for the given ImageView
		image.setFitWidth(dimension);
		// preserve the image ratio
		image.setPreserveRatio(true);
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}