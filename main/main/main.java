package main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.jfree.ui.RefineryUtilities;

public class main {
	
	static int features = 21;
	static int output = 3;
	static int hidden = 50;

	public static void main(String[] args) {

		// Hyper parameters
		int epochs = 200;
		int trainSize = 0;
		double learningRate = 0.2;

		dataRead dr = new dataRead(features, output);
		LinkedList<double[]> train = dr.readexcel(); // train [2126, 24]
		ArrayList<double[]> test = new ArrayList<double[]>();
		randomSelect(train, test); // train [1701, 24], test [425, 24]

		double[][] trainX = toMatrix(train);
		double[][] trainY = xySplit(trainX); // trainX [1701, 21], trainY [1701, 3]
		double[][] testX = toMatrix(test);
		double[][] testY = xySplit(testX); // testX [425, 21], testY [425, 3]

		double[] lossList = new double[epochs];
		double[] accuracyList = new double[epochs];
		network network = new network(features, hidden, output);

		for (int i = 0; i < epochs; i++) {
			// Forward
			accuracyList[i] = network.accuracy(trainX, trainY);
			
			// Calculate gradient and Update parameters
			network.gradient(trainX, trainY, learningRate);

			// Record
			double loss = network.loss(trainX, trainY);
			lossList[i] = loss;
		}
		for (int i = 0; i < epochs; i++) {
			System.out.println(lossList[i]);
		}
		graph(lossList);
	}
	
	public static void graph(double[] list) {
		graph g = new graph("FNN loss changes", list);
        g.pack();
        RefineryUtilities.centerFrameOnScreen(g);
        g.setVisible(true);
	}
	
	public static double[][] xySplit(double[][] x) {
		double[][] y = new double[x.length][output];
		double[] lineX, lineY;
		for (int i = 0; i < x.length; i++) {
			lineX = new double[features];
			lineY = new double[output];
			for (int j = 0; j < features; j++)
				lineX[j] = x[i][j];
			for (int j = 0; j < output; j++)
				lineY[j] = x[i][j+features];
			x[i] = lineX.clone();
			y[i] = lineY.clone();
		}
		return y;
	}
	
	public static <T> double[][] toMatrix(List<double[]> list) {
		double[][] mat = new double[list.size()][list.get(0).length];
		for (int i = 0; i < mat.length; i++)
			mat[i] = list.get(i).clone();
		return mat;
	}
	
	public static void randomSelect(LinkedList<double[]> mat, ArrayList<double[]> test) {
		for (int i = 0; i < 425; i++) {
			Collections.shuffle(mat);
			test.add(mat.removeLast());
		}
	}
}