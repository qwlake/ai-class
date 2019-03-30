package main;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;

public class graph extends ApplicationFrame{
	public graph(String title, double[] list) {
        super(title);
        DefaultCategoryDataset data = new DefaultCategoryDataset();
        
        int a = 0;
		for (double d : list) {
			data.addValue(Math.pow(d,2), "", "" + ++a);
		}
		
        JFreeChart chart = ChartFactory.createLineChart(
            "FNN loss change",
            "epochs", 
            "loss", 
            data,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 500));
        setContentPane(chartPanel);
    }
}
