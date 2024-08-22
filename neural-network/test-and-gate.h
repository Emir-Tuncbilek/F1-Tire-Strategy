//
// Created by Emir Tuncbilek on 7/21/24.
//

#ifndef F1_STRATEGIES_TEST_AND_GATE_H
#define F1_STRATEGIES_TEST_AND_GATE_H


#include "matrix.h"
inline std::vector<Matrix> genXValues() {
    std::vector<Matrix> xVals(16);
    {
        std::vector<std::unique_ptr<std::vector<double>>> vec(2);
        vec[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[0] = Matrix(std::move(vec));

        std::vector<std::unique_ptr<std::vector<double>>> vec1(2);
        vec1[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec1[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        xVals[1] = Matrix(std::move(vec1));

        std::vector<std::unique_ptr<std::vector<double>>> vec2(2);
        vec2[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec2[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        xVals[2] = Matrix(std::move(vec2));

        std::vector<std::unique_ptr<std::vector<double>>> vec3(2);
        vec3[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec3[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[3] = Matrix(std::move(vec3));

        std::vector<std::unique_ptr<std::vector<double>>> vec4(2);
        vec4[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec4[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[4] = Matrix(std::move(vec4));

        std::vector<std::unique_ptr<std::vector<double>>> vec5(2);
        vec5[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec5[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[5] = Matrix(std::move(vec5));

        std::vector<std::unique_ptr<std::vector<double>>> vec6(2);
        vec6[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec6[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[6] = Matrix(std::move(vec6));

        std::vector<std::unique_ptr<std::vector<double>>> vec7(2);
        vec7[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec7[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[7] = Matrix(std::move(vec7));

        std::vector<std::unique_ptr<std::vector<double>>> vec8(2);
        vec8[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec8[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        xVals[8] = Matrix(std::move(vec8));

        std::vector<std::unique_ptr<std::vector<double>>> vec9(2);
        vec9[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec9[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        xVals[9] = Matrix(std::move(vec9));

        std::vector<std::unique_ptr<std::vector<double>>> vec10(2);
        vec10[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec10[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[10] = Matrix(std::move(vec10));

        std::vector<std::unique_ptr<std::vector<double>>> vec11(2);
        vec11[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec11[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        xVals[11] = Matrix(std::move(vec11));

        std::vector<std::unique_ptr<std::vector<double>>> vec12(2);
        vec12[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec12[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[12] = Matrix(std::move(vec12));

        std::vector<std::unique_ptr<std::vector<double>>> vec13(2);
        vec13[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec13[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[13] = Matrix(std::move(vec13));

        std::vector<std::unique_ptr<std::vector<double>>> vec14(2);
        vec14[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec14[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[14] = Matrix(std::move(vec14));

        std::vector<std::unique_ptr<std::vector<double>>> vec15(2);
        vec15[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec15[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[15] = Matrix(std::move(vec15));

        /*
        std::vector<std::unique_ptr<std::vector<double>>> vec16(2);
        vec16[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec16[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        xVals[16] = Matrix(std::move(vec16));
         */
    }

    return xVals;
}

inline std::vector<Matrix> genYValues() {
    // 1, 0 is false, 0, 1 is true
    std::vector<Matrix> yVals(16);
    {
        std::vector<std::unique_ptr<std::vector<double>>> vec(2);
        vec[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[0] = Matrix(std::move(vec));  // True

        std::vector<std::unique_ptr<std::vector<double>>> vec1(2);
        vec1[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec1[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[1] = Matrix(std::move(vec1)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec2(2);
        vec2[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec2[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[2] = Matrix(std::move(vec2)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec3(2);
        vec3[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec3[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[3] = Matrix(std::move(vec3)); // True

        std::vector<std::unique_ptr<std::vector<double>>> vec4(2);
        vec4[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec4[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[4] = Matrix(std::move(vec4)); // True

        std::vector<std::unique_ptr<std::vector<double>>> vec5(2);
        vec5[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec5[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[5] = Matrix(std::move(vec5)); // True

        std::vector<std::unique_ptr<std::vector<double>>> vec6(2);
        vec6[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec6[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[6] = Matrix(std::move(vec6)); // True

        std::vector<std::unique_ptr<std::vector<double>>> vec7(2);
        vec7[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec7[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[7] = Matrix(std::move(vec7)); // True

        std::vector<std::unique_ptr<std::vector<double>>> vec8(2);
        vec8[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec8[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[8] = Matrix(std::move(vec8)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec9(2);
        vec9[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec9[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[9] = Matrix(std::move(vec9)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec10(2);
        vec10[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec10[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[10] = Matrix(std::move(vec10)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec11(2);
        vec11[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec11[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[11] = Matrix(std::move(vec11)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec12(2);
        vec12[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec12[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[12] = Matrix(std::move(vec12)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec13(2);
        vec13[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec13[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[13] = Matrix(std::move(vec13)); // True

        std::vector<std::unique_ptr<std::vector<double>>> vec14(2);
        vec14[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec14[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[14] = Matrix(std::move(vec14)); // False

        std::vector<std::unique_ptr<std::vector<double>>> vec15(2);
        vec15[0] = std::make_unique<std::vector<double>>(std::vector<double>{1}),
        vec15[1] = std::make_unique<std::vector<double>>(std::vector<double>{0});
        yVals[15] = Matrix(std::move(vec15)); // False

        /*
        std::vector<std::unique_ptr<std::vector<double>>> vec16(2);
        vec16[0] = std::make_unique<std::vector<double>>(std::vector<double>{0}),
        vec16[1] = std::make_unique<std::vector<double>>(std::vector<double>{1});
        yVals[16] = Matrix(std::move(vec16)); // True
         */
    }

    return yVals;
}

#endif //F1_STRATEGIES_TEST_AND_GATE_H
