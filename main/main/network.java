package main;

import java.util.LinkedHashMap;
import java.util.Random;

public class network {
	int ipSize;
	int hdSize;
	int opSize;
	LinkedHashMap<String, double[][]> params;
	double wtConst = 0.01;
	public network(int inputSize, int hiddenSize, int outputSize) {
		this.ipSize = inputSize;
		this.hdSize = hiddenSize;
		this.opSize = outputSize;
		double[][] w1 = new double[inputSize][hiddenSize];
		double[][] w2 = new double[hiddenSize][outputSize];
		double[][] b1 = new double[1][hiddenSize];
		double[][] b2 = new double[1][outputSize];
		this.params = new LinkedHashMap<String, double[][]>();
		
		Random r = new Random();
		for (int i = 0; i < w1.length; i++)
			for (int j = 0; j < w1[0].length; j++)
				w1[i][j] = wtConst * r.nextGaussian();
		r = new Random();
		for (int i = 0; i < w2.length; i++)
			for (int j = 0; j < w2[0].length; j++)
				w2[i][j] = wtConst * r.nextGaussian();
		for (int i = 0; i < b1[0].length; i++)
			b1[0][i] = 0;
		for (int i = 0; i < b2[0].length; i++)
			b2[0][i] = 0;
		params.put("w1", w1);
		params.put("b1", b1);
		params.put("w2", w2);
		params.put("b2", b2);
	}
	
	public double[][] predict(double[][] x) {
		double[][] a1 = dot(x, params.get("w1"), params.get("b1"));
		sigmoid(a1);
		double[][] a2 = dot(a1, params.get("w2"), params.get("b2"));
		sigmoid(a2);
		return a2;
	}
	
	public double loss(double[][] x, double[][] t) {
		double[][] y = predict(x);
		return meanSquared(y, t);
	}
	
	public double accuracy(double[][] x, double[][] t) {
		double[][] y = predict(x);
		int cnt = 0;
		for (int i = 0; i < y.length; i++)
			for (int j = 0; j < y[0].length; j++)
				cnt = (y[i][j] == 1 && t[i][j] == 1)? (cnt+1):cnt;
		return (double)cnt/y.length;
	}
	
	public LinkedHashMap<String, double[][]> gradient(double[][] x, double[][] t) {
		LinkedHashMap<String, double[][]> grads = new LinkedHashMap<String, double[][]>();
		grads.put("w1", differential(x,t));
		grads.put("w1", differential(x,t));
		grads.put("b2", differential(x,t));
		grads.put("b2", differential(x,t));
		return grads;
	}
	
	public double[][] differential(double[][] x, double[][] t) {
		double h = 1e-4;	// 0.0001
		double[][] grad = new double[x.length][x[0].length];
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[0].length; j++) {
				double temp = x[i][j];
				x[i][j] = temp + h;
				double fxh1 = loss(x, t);
				x[i][j] = temp - h;
				double fxh2 = loss(x, t);
				grad[i][j] = (fxh1 - fxh2) / (2*h);
				x[i][j] = temp;
			}
		}
		return grad;
	}
	
	public double[][] dot(double[][] x, double[][] w, double[][] b) {
		double[][] a = new double[x.length][w[0].length];
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[0].length; j++)
				a[i][j] = b[0][j];	// Init the matrix and Add bias
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < w[0].length; j++)
				for (int k = 0; k < x[0].length; k++)
					a[i][j] += x[i][k] * w[k][j];
		return a;
	}
	
	public double meanSquared(double[][] y, double[][] t) {
		double sum = 0;
		for (int i = 0; i < y.length; i++)
			for (int j = 0; j < y[0].length; j++)
				sum += Math.pow(y[i][j]-t[i][j], 2);
		return 0.5 * sum;
	}
	
	public void sigmoid(double[][] x) {
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[0].length; j++)
				x[i][j] = 1 / (1 + Math.pow(Math.E, -x[i][j]));
	}
	
	public void shape(double[][] arr) {
		System.out.println("[" + arr.length + "," + arr[0].length + "]");
	}
	
	public void print(double[][] arr) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[0].length; j++)
				System.out.print(arr[i][j]);
			System.out.println();
		}
	}
}
