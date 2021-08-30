LightGBM Transform Tutorial
===========================

The purpose of this document is to give you a tutorial on how to use transform in LightGBM.

Transformation is a process to convert data/feature from one format to another.
Now we support two kinds of transformations in LightGBM:

-   Linear. Linear transformation, could be adjusted by slope and intercept.

-   FreeForm2. FreeForm2 is a more flexible transform, created by Microsoft Core Ranking and used widely over Microsoft production model training.
    As the name indicates, FreeForm2 empowers users to compose a free combination of features as they like. It is expressed by formulas to be applied in the model inputs.
    The surface syntax is s-expression, with parentheses in a LISP-like fashion to delimit. 
    FreeForm2 has implicit type systems and evaluate a single, nested expression that returns a floating-point number.

How to use transform in LightGBM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
------------

See `Transform Installation <./Installation-Guide.rst#build-transform-version>`__, 
install dependencies and build LightGBM.

Data preparation
----------------
1.  Input data. Data file used for training or prediction.

    **Note**: only TSV is supported now.

    **Note**: the file is better to have a header, or you need to provide a separate header file instead.

2.  Transform file. The ini contains all real features and expressions following a specific format. Below is an example, you could see `FreeForm2 language spec <#freeform2-language-spec>`__ to learn more about the grammar.

    .. code::

        [Input:0]
        Name=m:QueryId
        Transform=Linear
        Slope=1
        Intercept=0

        [Input:1]
        Transform=FreeForm2
        Line1=(* feature_0 feature_1)

        [Input:2]
        Transform=FreeForm2
        Line1=(if (> feature_1 0) (- feature_2 feature_9) 1000000)

        ...

    **Note**: this file is not a supplement of raw features, but all used for training. Use "Linear" type if you want to keep the original ones.

    **Note**: transformed feature index ranges from 0 to the maximum "Input" value given in transform file.
    By default, will pad 0 as feature value for missing indices within the range.

3.  Header file. The file contains feature names.

    **Note**: Could skip if input data has header, or must be provided.

**Note**: if no transform file or header file is given, 
the input data will be used as features directly for training.

Run task with transform
-----------------------

Actually, the use way is the same as previous, as no interface changes of both CLI and python sdk.

Run the following command

::

    ./lightgbm config=lightgbm.conf data=path/to/train.tsv transform_file=path/to/transform.ini header_file=path/to/header.tsv


Or use python sdk

.. code::

    train_data = lgb.Dataset("path/to/train.tsv", params={"transform_file": "path/to/transform.ini", "header_file": "path/to/header.tsv"})
    valid_data = lgb.Dataset("path/to/valid.tsv", params={"transform_file": "path/to/transform.ini", "header_file": "path/to/header.tsv"})
    # train and predict.
    bst = lgb.train(params, train_data, valid_sets=[valid_data])
    pred = bst.predict("path/to/test.tsv")
    # save model.
    bst.save_model(trained_model_path)
    # load model and predict again.
    bst = lgb.Booster(model_file=trained_model_path)
    pred = bst.predict("path/to/test.tsv")

**Note**: transform file content and header line will be saved at the bottom of model file, 
with section flags "transform" and "header".

FreeForm2 language spec
~~~~~~~~~~~~~~~~~~~~~~~

Following is a brief introduction of FreeForm2 language.

The FreeForm2 type system consists of floating point numbers \(currently 32-bit\), signed integers \(currently 64-bit\), booleans, and multi-dimensional arrays of the basic quantities. The surface syntax of FreeForm2 is s-expression languages, which use parentheses in a LISP-like fashion to delimit expressions. It can evaluate a single, nested expression that returns a floating point number.

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Literals
--------

Examples of literal integers are 2 and -5435. Literal floating point numbers are 2.0, -0.234, -2E-10. The literal booleans are true and false. See the array operation section for an example of an array literal.

Feature References
------------------

As in the FreeForm language, references to feature names are allowed within FreeForm2. For example, NumberOfOccurrences\_Body\_0 will retrieve the feature of that name for use in the program. However, unlike the FreeForm language \(where they were floats\), features are integers in the FreeForm2 language are integers. This means that \(/ NumberOfOccurrences\_Body\_0 5\) will perform **integer** division, rather than floating point division. See the 'Type Conversion' section for information on converting between types.

Type Conversion
---------------

Integers can be converted to floating point numbers using the 'float' operator. Floating point numbers can be converted to integers using the 'truncate'/'int' and 'round' operators. Booleans can be converted to integers using the 'int' operator.

Examples
++++++++


* \(== \(float 4\) 4.0\)
* \(== \(int 4.5\) 4\)
* \(== \(truncate 4.5\) 4\)
* \(== \(round 4.5\) 5\)
* \(== \(int false\) 0\)
* \(== \(int true\) 1\)

Arithmetic operators
--------------------

FreeForm2 offers a familiar set of unary and binary operators. These operators act like operators in C and C++, in that they determine the types of their arguments, and convert the result to the appropriate type. In effect, operators with all integers for arguments will produce integers, and anything taking a float will produce a float \("floating point contagion"\). Note that integer arithmetic is currently signed 64-bit, and is subject to the C/C++ rules about arithmetic overflow and underflow \(in that underflow and overflow are silently allowed\). The unary operators are:

* \-: negation
* ln: natural logarithm. This operation does not support integers yet. All edge cases \(non-positive operands\) return negative infinity.
* ln1: \(ln1 x\) is equivalent to \(ln \(+ x 1.0\)\). This operation does not support integers yet. All edge cases \(non-positive operands\) return negative infinity.
* abs: absolute value

The binary operators are:

* +: addition
* \-: subtraction
* \*: multiplication
* /: division
* mod: modulus
* \*\*: power \(an alias of '\-'\)
* \-: power \(an alias of '\*\*'\)
* max: maximum
* min: minimum
* log: logarithm of a supplied operand and a base. This operation does not support integers yet. All edge cases \(non-positive operands\) return negative infinity.

The '+' operator is actually n-ary, which means it accepts any number of arguments. For example, \(+ 1 2 3 4 5 6\) is legal, and will produce the sum of all of its operands. The freeform language provides no guarantees of the order in which these operands are evaluated. This could be extended to many of the other operators, though it is unclear whether this is more useful than potentially confusing.

The FreeForm and FreeForm2 languages have a set of quirks, that define behaviour of arithmetic operators in what would normally be error conditions:

* All of the logarithm operators define the logarithm of a non-positive number to be zero.
* Division and modulus define the result of division/modulus by zero to be zero. In addition, division of the minimum integer by negative one \(usually a hardware exception condition, as there isn't room in two's-complement representation for this number\) to be the maximum integer, one less than it 'should' be. Division of the minimum integer by one resulted in undefined behaviour in the FreeForm language.

Examples
++++++++

Unary operators
'''''''''''''''

* \(== \(- -4.0\) 4.0\)
* \(== \(ln 10\) 2.30258\)
* \(== \(ln1 9\) 2.30258\)

Binary operators
''''''''''''''''

* \(== \(+ 5 5.0\) 10.0\)
* \(== \(+ 1 2 3 4 5 6\) 21\)
* \(== \(- 5 3\) 2\)
* \(== \(\* 3.3 4.4\) 14.52\)
* \(== \(/ 10 3\) 3\)
* \(== \(mod 10 3\) 1\)
* \(== \(\*\* 2 8\) 256\)
* \(== \(max 10 13.5\) 13.5\)
* \(== \(min -4 2\) -4\)
* \(== \(log 256.0 2\) 8.0\)
* \(== \(log -256.0 2\) 0.0\)

Quirks
''''''

* \(== \(/ -9223372036854775808 -1\) 9223372036854775807\)
* \(== \(/ 10 0.0\) 0.0\)
* \(== \(mod 10.0 0.0\) 0.0\)
* \(== \(ln -10.0\) -infinity\) \(note that '-infinity' is not a real constant in freeform2\)
* \(== \(log 10.0 0.0\) -infinity\) \(note that '-infinity' is not a real constant in freeform2\)
* \(== \(log 0.0 10.0\) -infinity\) \(note that '-infinity' is not a real constant in freeform2\)

Comparison and Logical Operators
--------------------------------

The usual set of comparison operators is available. They work over integers and floats, and require that the two operands be of the same type. The comparison operators are: '==', '\!=', '\<', '\<=', '>', '>='.

Examples
++++++++

* \(== \(\< 5 6\) true\)
* \(== \(> 5.5 6.0\) false\)

In addition, logical operators are available over boolean quantities, such as those produced by the comparison operators. The available operators are 'and', '\&\&', 'or', '||', and 'not'. 'and' and '\&\&' are synonymous, as are 'or' and '||'. 'not' is unary, where the other operators are binary.

Examples
++++++++

* \(== \(and \(\< 5 6\) \(== 1 1\)\) true\)
* \(== \(\&\& \(\< 5 6\) \(== 1 1\)\) true\)
* \(== \(or \(\< 5 6\) \(== 2 1\)\) true\)
* \(== \(not false\) true\)

Bitwise operators
-----------------

Bitwise operators are available over integer quantities. These are 'bitand', 'bitor', 'bitnot'. For example, \(bitand 7 1\) will evaluate to 1. 'bitnot' is unary, where the other bitwise operators are binary.

Examples
++++++++

* \(== \(bitand 5 3\) 1\)
* \(== \(bitor 5 3\) 7\)
* \(== \(bitnot 0\) -1\)

Let Binding
-----------

FreeForm2 allows a 'let' expression to bind quantities within the same NeuralInput to be bound to variable names. Note however, that no mutation of variables is allowed. The types of these quantities are determined from their definitions. 'let' expression take two operands: the first is a set of parenthesized pairs giving names and definitions, the second provides the value returned by the 'let' expression, which may reference the bound quantities. For example, \(let \(\(x 1\) \(y 2\)\) \(+ x y\)\) will evaluate to 3. Note that each subsequent binding given can refer to the previous one, so that \(let \(\(x 1\) \(y \(\* x x\)\)\) \(+ y x\)\) is legal.

Values bound by a let statement are only available within the scope of that let statement. There is currently no mechanism to bind values in the global namespace.

Example
+++++++

* \(== \(let \(\(x 3\) \(y \(+ x 2\)\)\) \(\* x y\)\) 15\)

Conditionals
------------

FreeForm2 provides an 'if' statement, taking three arguments. The first must be boolean, and dictates which of the other two statements is evaluated. The remaining two arguments must be of the same type, which will be the result type of the conditional. For example, \(if \(> 1 4\) 5.4 6.7\) will evaluate to 6.7 \(the 'else' branch\). In addition to straight conditionals, there is also a 'select-nth' expression. This allows selection of a subexpression by index. The first argument provides an integer index. 'select-nth' will then accept any number of operands of the same type, which will be selected depending on the value of the first operand, indexed from zero. For example, \(select-nth 1 10 11 12 13\) will evaluate to 11. \(select-nth 3 10 11 12 13\) will evaluate to 13. Any out-of-bounds index provided will select the 'zero' value that is appropriate for the expression \(0, 0.0, false, or \[\]\).

Examples
++++++++

* \(== \(if true 0 1\) 0\)
* \(== \(if \(\< 3 5\) 0.5 1.0\) 1.0\)
* \(== \(select-nth \(+ 1 2\) 0.1 0.2 0.3 0.4 0.5\) 0.4\)
* \(== \(select-nth -1 0.1 0.2 0.3 0.4 0.5\) 0.1\)
* \(== \(select-nth 100000 0.1 0.2 0.3 0.4 0.5\) 0.5\)

Loops
-----

FreeForm2 provides a limited looping construct, 'range-reduce'. This allows you to loop over ranges of integers. The arguments to range-reduce are:

1.  the range variable \(i.e. 'i', 'index'\)
2.  the integer lower bound of the loop
3.  the integer upper bound of the loop \(not inclusive, so this bound will not actually be reached\)
4.  the accumulator variable, which can be of any type, and stores the current result of the reduction
5.  the initial accumulator value
6.  the reduction expression, which must be of the same type as the initial accumulator value

Examples
++++++++

* \(== \(range-reduce i 0 10 acc 0 \(+ i acc\)\) 45\)
* \(== \(let \(\(arr \(array-literal \[1 22 3\]\)\)\) \(range-reduce i 0 \(array-length arr\) acc 0 \(max arr\[i\] acc\)\) 22\)

The first example above performs a reduction over integers in the range 0 to 10, with the current value kept in variable 'i'. The accumulator variable \('acc'\) starts with value 0. In the first iteration, both acc and i are 0, so the reduction evaluates to 0, which is stored back in acc. In the second iteration, i becomes 1, and so the expression evaluates to 1, which is stored in acc. In the third iteration, i is 2, acc becomes 3, and so on until we calculate the same 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9. The second example uses a range-reduce expression to calculate the maximum element in an array of integers. Note that the range-reduce expression will return the initial value given to the accumulator variable in cases where it loops zero times. If the upper bounds given is less than the lower bound given, then range-reduce will loop zero times.

Array operations
----------------

FreeForm2 provides a number of operations to manipulate arrays. An array-literal expression allows creation of arrays. The length of an array can be retrieved by the array-length operation. Array values can be accessed using the familiar '\[index\]' post-fix notation, as seen in C and C++. Array accesses to out-of-bounds locations will resolve to 0, 0.0, or false, depending on the type of the array.

Examples
++++++++

* \(array-literal \[0 1 2\]\)
* \(== \(array-literal \[0 1 2\]\)\[1\] 1\)
* \(== \(array-literal \[\[0 1 2\] \[3 4 5\]\]\)\[1\] \(array-literal \[3 4 5\]\)\)
* \(== \(array-literal \[\[0 1 2\] \[3 4 5\]\]\)\[1\]\[2\] 5\)
* \(== \(array-length \[\[0 1 2\] \[3 4 5\]\]\) 2\)
* \(== \(array-length \[0 1 2\]\) 3\)

Quirks
''''''

* \(== \(array-literal \[1 2 3\]\)\[-1\] 0\)
* \(== \(array-literal \[1 2 3\]\)\[10000\] 0\)
* \(== \(array-literal \[\] int\)\[0\] 0\)
* \(== \(array-literal \[\] float\)\[0\] 0.0\)
* \(== \(array-literal \[\] bool\)\[0\] false\)
