package weka.filters.unsupervised.instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.filters.SimpleBatchFilter;

public class Smear extends SimpleBatchFilter implements Randomizable {



    @Override
    public void setSeed(int seed) {

    }

    @Override
    public int getSeed() {
        return 0;
    }

    @Override
    public String globalInfo() {
        return null;
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return null;
    }

    @Override
    protected Instances process(Instances instances) throws Exception {
        return null;
    }
}
