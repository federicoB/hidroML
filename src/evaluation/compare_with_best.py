# given a model name load it, compare it with current best
import argparse
from src.evaluation.evaluate_model import evaluate_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model', default='best')
    parser.add_argument('--new_model',  default='new')
    args = vars(parser.parse_args())

    best_model_name = args['best_model']
    new_model_name = args['new_model']
    old_results = evaluate_model(best_model_name)
    new_results = evaluate_model(new_model_name)
    print(old_results)
    print(new_results)
    percents = [((new_results[i]-old_results[i])/old_results[i])*100 for i in range(3)]
    output = "{} has {:.2f}% the parameters and {:.2f}% the operations of {} and scores {:.2f}% more prediction error"\
        .format(new_model_name,percents[0],percents[1],best_model_name,percents[2])
    print(output)
