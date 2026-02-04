from log_code import setup_logging
logger = setup_logging('variable_transformation')

import sys
import numpy as np
from scipy import stats
from scipy.stats import skew


# ======================================================
# TRANSFORMATION FUNCTIONS (ONE PER METHOD)
# ======================================================

def log_transform(x_train_num, x_test_num, col):
    x_train_num[col + '_log'] = np.log1p(x_train_num[col])
    x_test_num[col + '_log'] = np.log1p(x_test_num[col])
    return x_train_num, x_test_num, col + '_log'


def sqrt_transform(x_train_num, x_test_num, col):
    x_train_num[col + '_sqrt'] = np.sqrt(x_train_num[col])
    x_test_num[col + '_sqrt'] = np.sqrt(x_test_num[col])
    return x_train_num, x_test_num, col + '_sqrt'


def reciprocal_transform(x_train_num, x_test_num, col):
    x_train_num[col + '_reciprocal'] = 1 / (x_train_num[col] + 1e-6)
    x_test_num[col + '_reciprocal'] = 1 / (x_test_num[col] + 1e-6)
    return x_train_num, x_test_num, col + '_reciprocal'


def exp_transform(x_train_num, x_test_num, col):
    x_train_num[col + '_exp'] = np.exp(x_train_num[col].clip(upper=10))
    x_test_num[col + '_exp'] = np.exp(x_test_num[col].clip(upper=10))
    return x_train_num, x_test_num, col + '_exp'


def boxcox_transform(x_train_num, x_test_num, col):
    x_train_num[col + '_boxcox'], lam = stats.boxcox(x_train_num[col])
    x_test_num[col + '_boxcox'] = stats.boxcox(x_test_num[col], lmbda=lam)
    return x_train_num, x_test_num, col + '_boxcox'


def yeojohnson_transform(x_train_num, x_test_num, col):
    x_train_num[col + '_yeo'], lam = stats.yeojohnson(x_train_num[col])
    x_test_num[col + '_yeo']= stats.yeojohnson(x_test_num[col], lmbda=lam)
    return x_train_num, x_test_num, col + '_yeo'


# ======================================================
# MAIN VARIABLE TRANSFORMATION FUNCTION (REFERENCE STYLE)
# ======================================================

def vt(x_train_num, x_test_num):
    try:
        for col in x_train_num.columns:

            data = x_train_num[col]

            # ---------------- FIND BEST METHOD USING SKEWNESS ----------------
            skew_dict = {}

            skew_dict['original'] = abs(skew(data))
            skew_dict['log'] = abs(skew(np.log1p(data)))
            skew_dict['reciprocal'] = abs(skew(1 / (data + 1e-6)))
            skew_dict['exp'] = abs(skew(np.exp(data.clip(upper=10))))

            if (data >= 0).all():
                skew_dict['sqrt'] = abs(skew(np.sqrt(data)))

            if (data > 0).all():
                bc, _ = stats.boxcox(data)
                skew_dict['boxcox'] = abs(skew(bc))

            yj, _ = stats.yeojohnson(data)
            skew_dict['yeo'] = abs(skew(yj))

            best_method = min(skew_dict, key=skew_dict.get)
            logger.info(f'Best transformation for {col} : {best_method}')

            # ---------------- APPLY BEST METHOD ----------------
            if best_method == 'log':
                x_train_num, x_test_num, new_col = log_transform(x_train_num, x_test_num, col)

            elif best_method == 'sqrt':
                x_train_num, x_test_num, new_col = sqrt_transform(x_train_num, x_test_num, col)

            elif best_method == 'reciprocal':
                x_train_num, x_test_num, new_col = reciprocal_transform(x_train_num, x_test_num, col)

            elif best_method == 'exp':
                x_train_num, x_test_num, new_col = exp_transform(x_train_num, x_test_num, col)

            elif best_method == 'boxcox':
                x_train_num, x_test_num, new_col = boxcox_transform(x_train_num, x_test_num, col)

            elif best_method == 'yeo':
                x_train_num, x_test_num, new_col = yeojohnson_transform(x_train_num, x_test_num, col)

            else:
                continue

            # ---------------- DROP ORIGINAL COLUMN ----------------
            x_train_num = x_train_num.drop(col, axis=1)
            x_test_num = x_test_num.drop(col, axis=1)

        logger.info(f'Final transformed columns : {x_train_num.columns}')
        return x_train_num, x_test_num

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line no : {er_line.tb_lineno} due to {er_msg}')

