package weka.filters.unsupervised.instance;
import no.uib.cipr.matrix.*;
import org.apache.commons.math3.distribution.BetaDistribution;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

import java.util.Random;

public class Mixup extends SimpleBatchFilter implements Randomizable {
    /**
     * The seed for the random number generator.
     */
    protected int m_seed = 0;

    /**
     * The number of instances to output per input instance.
     */
    protected int numSamples = 1;

    /**
     * The value of alpha to use in the beta distribution
     */
    protected double m_alpha = 0.05;

    /**
     * The value of lambda for weighing the mixing and the resulting instances
     */
    protected double m_lambda;


    @Override
    public void setSeed(int seed) {
        this.m_seed = seed;
    }

    @Override
    public int getSeed() {
        return m_seed;
    }

    public int getNumSamples() {
        return numSamples;
    }

    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    public double getAlpha() {
        return m_alpha;
    }

    public void setAlpha(double alpha) {
        this.m_alpha = alpha;
    }

    @Override
    public String globalInfo() {
        return "This filter implements the 'Mixup' approach to generate variable data from an initial dataset";
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        if (m_Debug) {
            System.err.println("Determining output format.");
        }
        Instances output = new Instances(inputFormat, 0);
//        output.setClassIndex(output.numAttributes() - 1);
        if (m_Debug) {
            System.err.println("Finished determining output format with " + output.numAttributes() + " attributes.");
        }
        return output;    }


    /**
     * The method that processes the given dataset and outputs the filtered data.
     *
     * @param instances the input data to be filtered
     * @return the filtered data, consisting of randomly mixed and weighted pairs of instances ready for other machine learning algorithms
     */
    @Override
    protected Instances process(Instances instances) throws Exception {
        Instances output = getOutputFormat();

        Random rand = new Random(m_seed);
        BetaDistribution beta = new BetaDistribution(m_alpha, m_alpha);

        m_lambda = beta.inverseCumulativeProbability(Math.random()); // beta distribution via apache commons math3


        for (int i = 0; i < instances.numInstances(); i++){
            // Select the next instance, then randomly select a second instance for mixing
            // should also try randomly selecting both instances.
            for(int j = 0; j < numSamples; j++) {
                Instance firstInputInstance = instances.instance(i);
                Instance secondInputInstance = instances.instance(rand.nextInt(instances.numInstances()));

                // turn instance attributes into vectors for multiplication and addition
                Vector firstInputAttributes = new DenseVector(firstInputInstance.toDoubleArray());
                Vector secondInputAttributes = new DenseVector(secondInputInstance.toDoubleArray());

                // scale attribute values to lambda an 1 - lambda
                firstInputAttributes = firstInputAttributes.scale(m_lambda);
                secondInputAttributes = secondInputAttributes.scale(1 - m_lambda);

                // add scaled attributes and
                double[] outputAttributes = ((DenseVector) firstInputAttributes.add(secondInputAttributes)).getData();

                // add mixed attributes to two new instances, with the given weights and class values
                Instance firstOutputInstance = new DenseInstance(m_lambda, outputAttributes);
                Instance secondOutputInstance = new DenseInstance(1 - m_lambda, outputAttributes);
                firstOutputInstance.setDataset(instances);
                secondOutputInstance.setDataset(instances);
                firstOutputInstance.setClassValue(firstInputInstance.classValue());
                secondOutputInstance.setClassValue(secondInputInstance.classValue());
                output.add(firstOutputInstance);
                output.add(secondOutputInstance);
            }
        }
        return output;
    }
}
