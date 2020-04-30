package weka.filters.unsupervised.instance;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import org.apache.commons.math3.distribution.BetaDistribution;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.filters.SimpleBatchFilter;

import java.util.*;
import java.util.stream.Collectors;

public class Smear extends SimpleBatchFilter implements Randomizable {

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
    protected double m_stdDev = 0.05;

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

    public double getStdDev() {
        return m_stdDev;
    }

    public void setStdDev(double m_stdDev) {
        this.m_stdDev = m_stdDev;
    }

    @Override
    public String globalInfo() {
        return "This filter implements the 'Smear' approach to generate variable data from an initial dataset";
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

    @Override
    protected Instances process(Instances instances) throws Exception {
//        Instances output = getOutputFormat();
        Random rand = new Random(m_seed);
        int k_gap = 10;
        double attr_multiplier;
        double gaussian;

        for (int i = 0; i < instances.numAttributes(); i++) {
            // Get instances attribute values
            double[] initialAttributeValues = instances.attributeToDoubleArray(i);
            // Sort values and remove duplicates with TreeSet
            TreeSet<Double> sortedAttributeValues = Arrays.stream(initialAttributeValues).boxed().collect(Collectors.toCollection(TreeSet::new));
            // Set up new TreeSet to hold sorted differences
            SortedSet<Double> sortedDifferences = new TreeSet<>();
            // get number of distinct attributes and loop through, adding the difference between
            // consecutive values to differences set
            int numSortedAttributeVals = sortedAttributeValues.size();
            for (int j = 0; j < numSortedAttributeVals - 1; j++) {
                double lowestRemaining = sortedAttributeValues.pollFirst();
                double nextLowestRemaining = sortedAttributeValues.higher(lowestRemaining);
                sortedDifferences.add(nextLowestRemaining - lowestRemaining);
            }
            // Get kth (10th) smallest difference from Set, after turning into array
            Double[] differencesArray = sortedDifferences.toArray(new Double[0]);
            if (differencesArray.length >= k_gap){
                attr_multiplier = differencesArray[k_gap-1];
            } else {
                attr_multiplier = differencesArray[differencesArray.length-1];
            }

            // Replace current attribute values with perturbed values
            for (int i1 = 0; i1 < instances.size(); i1++) {
                gaussian = rand.nextGaussian() * (m_stdDev * attr_multiplier);
                // oh will this essentailly create a new instance not part of the set? prob need to pull out
                Instance updatedInstance = instances.instance(i1);
                updatedInstance.setValue(i, initialAttributeValues[i] + gaussian);
                instances.set(i1,updatedInstance);
            }
        }



//        for (int i = 0; i < instances.numInstances(); i++){
//            // Select the next instance, then randomly select a second instance for mixing
//            // should also try randomly selecting both instances.
//            for(int j = 0; j < numSamples; j++) {
//                Instance firstInputInstance = instances.instance(i);
//                Instance secondInputInstance = instances.instance(rand.nextInt(instances.numInstances()));
//
//                // turn instance attributes into vectors for multiplication and addition
//                Vector firstInputAttributes = new DenseVector(firstInputInstance.toDoubleArray());
//                Vector secondInputAttributes = new DenseVector(secondInputInstance.toDoubleArray());
//
//                // scale attribute values to lambda an 1 - lambda
//                firstInputAttributes = firstInputAttributes.scale(m_stdDev);
//                secondInputAttributes = secondInputAttributes.scale(1 - m_stdDev);
//
//                // add scaled attributes and
//                double[] outputAttributes = ((DenseVector) firstInputAttributes.add(secondInputAttributes)).getData();
//
//                // add mixed attributes to two new instances, with the given weights and class values
//                Instance firstOutputInstance = new DenseInstance(m_stdDev, outputAttributes);
//                Instance secondOutputInstance = new DenseInstance(1 - m_stdDev, outputAttributes);
//                firstOutputInstance.setDataset(instances);
//                secondOutputInstance.setDataset(instances);
//                firstOutputInstance.setClassValue(firstInputInstance.classValue());
//                secondOutputInstance.setClassValue(secondInputInstance.classValue());
//                output.add(firstOutputInstance);
//                output.add(secondOutputInstance);


        return instances;
    }

}
