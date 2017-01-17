module linalg.matrix;

import std.algorithm: min, max;
import std.array: join, front, empty;
import std.conv: to;
import std.numeric: dotProduct;
import std.range: ElementType;
import std.string: rightJustify;
import std.traits: CommonType, isNumeric, isInstanceOf;

immutable size_t DYNAMIC = 0;

private immutable
{
  auto op_index_one = q{
    ref opIndex(in size_t ind)
    {
      static if (r == 1)
      {
        return data[0][ind];
      }
      else
      {
        return data[ind][0];
      }
    }
  };

  auto op_index_two = q{
    ref opIndex(in size_t row, in size_t col)
    {
      return data[row][col];
    }
  };
}

alias SquareMatrix(type, size_t size) = Matrix!(type, size, size);
alias Vector(type, size_t size = DYNAMIC) = Matrix!(type, size, 1);
alias RowVector(type, size_t size = DYNAMIC) = Matrix!(type, 1, size);
alias isMatrix(type) = isInstanceOf!(Matrix, type);

// Aliases Matrix {{{1
alias MatrixXi = Matrix!int;
alias MatrixXf = Matrix!float;
alias MatrixXd = Matrix!double;
alias MatrixXr = Matrix!real;

alias Matrix2i = SquareMatrix!(int, 2);
alias Matrix2f = SquareMatrix!(float, 2);
alias Matrix2d = SquareMatrix!(double, 2);
alias Matrix2r = SquareMatrix!(real, 2);

alias Matrix3i = SquareMatrix!(int, 3);
alias Matrix3f = SquareMatrix!(float, 3);
alias Matrix3d = SquareMatrix!(double, 3);
alias Matrix3r = SquareMatrix!(real, 3);

alias Matrix4i = SquareMatrix!(int, 4);
alias Matrix4f = SquareMatrix!(float, 4);
alias Matrix4d = SquareMatrix!(double, 4);
alias Matrix4r = SquareMatrix!(real, 4);

// Aliases Vector {{{1
alias VectorXi = Vector!int;
alias VectorXf = Vector!float;
alias VectorXd = Vector!double;
alias VectorXr = Vector!real;

alias Vector2i = Vector!(int, 2);
alias Vector2f = Vector!(float, 2);
alias Vector2d = Vector!(double, 2);
alias Vector2r = Vector!(real, 2);

alias Vector3i = Vector!(int, 3);
alias Vector3f = Vector!(float, 3);
alias Vector3d = Vector!(double, 3);
alias Vector3r = Vector!(real, 3);

alias Vector4i = Vector!(int, 4);
alias Vector4f = Vector!(float, 4);
alias Vector4d = Vector!(double, 4);
alias Vector4r = Vector!(real, 4);

// Aliases RowVector {{{1
alias RowVectorXi = RowVector!int;
alias RowVectorXf = RowVector!float;
alias RowVectorXd = RowVector!double;
alias RowVectorXr = RowVector!real;

alias RowVector2i = RowVector!(int, 2);
alias RowVector2f = RowVector!(float, 2);
alias RowVector2d = RowVector!(double, 2);
alias RowVector2r = RowVector!(real, 2);

alias RowVector3i = RowVector!(int, 3);
alias RowVector3f = RowVector!(float, 3);
alias RowVector3d = RowVector!(double, 3);
alias RowVector3r = RowVector!(real, 3);

alias RowVector4i = RowVector!(int, 4);
alias RowVector4f = RowVector!(float, 4);
alias RowVector4d = RowVector!(double, 4);
alias RowVector4r = RowVector!(real, 4);
// }}}

/**
 * NOTES.
 * 1. ref opAssign(spec : type)(in spec[][] b) not work on static columns
 * 2. Если возможно, передаем по ссылке (не выделяя память). Компилятор сам выберет "правильный" метод.
 */

struct Matrix(type, size_t r = DYNAMIC, size_t c = DYNAMIC) if (isNumeric!type)
{
  // Aliases {{{1
  private
  {
    alias selftype = typeof(this);

    static if (r == DYNAMIC && c == DYNAMIC)
    {
      alias datatype = type[][];
    }
    else static if (r == DYNAMIC)
    {
      alias datatype = type[c][];
    }
    else static if (c == DYNAMIC)
    {
      alias datatype = type[][r];
    }
    else
    {
      alias datatype = type[c][r];
    }
  }
  // }}}

  datatype data;

  pure nothrow @safe
  {
    /++
     + При наличии this(...)(... Matrix!(...) ...) копирующий конструктор
     + не нужен
     +/
    // Constructors {{{1
    this(spec : type, size_t br, size_t bc)(in auto ref Matrix!(spec, br, bc) b)
    {
      opAssign(b.data);
    }

    this(spec)(in spec[] b) if (is (ElementType!spec : type))
    {
      opAssign(b);
    }

    static if (r == DYNAMIC && c == DYNAMIC)
    {
      this(in size_t rows, in size_t cols)
      {
        init(rows, cols);
      }
    }
    else static if (r == DYNAMIC || c == DYNAMIC)
    {
      this(in size_t len)
      {
        init(len);
      }
    }

    static if (r == 1 || c == 1)
    {
      this(spec : type)(in spec[] b)
      {
        opAssign(b);
      }
    }

    // Public Methods {{{1
    ref set_constant(spec : type)(in auto ref spec val)
    {
      foreach (ref row; data)
      {
        row[] = val;
      }

      return this;
    }

    ref set_zero()
    {
      return set_constant(0);
    }

    ref set_ones()
    {
      return set_constant(1);
    }

    ref set_diagonal(spec : type)(in auto ref spec val)
    {
      set_zero();

      const auto min = min(rows, cols);

      foreach (i; 0 .. min)
      {
        data[i][i] = val;
      }

      return this;
    }

    ref set_identity()
    {
      return set_diagonal(1);
    }

    ref do_triangular_view()
    {
      return this = triangular_view;
    }

    ref do_diagonal_view()
    {
      return this = diagonal_view;
    }

    ref transpose_in_place()
    {
      return this = transpose;
    }

    static if (r == DYNAMIC && c == DYNAMIC)
    {
      ref init(in size_t rows, in size_t cols)
      {
        return this = new datatype(rows, cols);
      }

      ref init_constant(spec : type)(in size_t rows, in size_t cols, in auto ref spec val)
      {
        return this = constant(rows, cols, val);
      }

      ref init_zero(in size_t rows, in size_t cols)
      {
        return this = zero(rows, cols);
      }

      ref init_ones(in size_t rows, in size_t cols)
      {
        return this = ones(rows, cols);
      }

      ref init_diagonal(spec : type)(in size_t rows, in size_t cols, in auto ref spec val)
      {
        return this = diagonal(rows, cols, val);
      }

      ref init_identity(in size_t rows, in size_t cols)
      {
        return this = identity(rows, cols);
      }
    }
    else static if (r == DYNAMIC || c == DYNAMIC)
    {
      ref init(in size_t len)
      {
        static if (r == DYNAMIC)
        {
          this = new datatype(len);
        }
        else
        {
          foreach (ref row; data)
          {
            row = new type[len];
          }
        }

        return this;
      }

      ref init_constant(spec : type)(in size_t len, in auto ref spec val)
      {
        return this = constant(len, val);
      }

      ref init_zero(in size_t len)
      {
        return this = zero(len);
      }

      ref init_ones(in size_t len)
      {
        return this = ones(len);
      }

      ref init_diagonal(spec : type)(in size_t len, in auto ref spec val)
      {
        return this = diagonal(len, val);
      }

      ref init_identity(in size_t len)
      {
        return this = identity(len);
      }
    }
    // }}}
  }

  pure nothrow
  {
    // Assignment Operator Overloading {{{1
    ref opAssign(spec : type, size_t br, size_t bc)(in auto ref Matrix!(spec, br, bc) b)
    {
      return this = b.data;
    }

    ref opAssign(spec)(in spec[] b) if (is (ElementType!spec : type))
    {
      static if (r == DYNAMIC)
      {
        data.length = b.length;
      }
      else
      {
        assert(r == b.length, "Error. this.r != b.length.");
      }

      foreach (i, ref row; data)
      {
        /** DMD Bag. Statement is not reachable
        static if (c != DYNAMIC)
        {
          assert(c == b[i].length, "Error. this.c != b[i].length.");
        }
        */

        // If incorrect length: lengths don't match for array copy
        row = to!(type[])(b[i]);
      }

      return this;
    }

    static if (r == 1 || c == 1)
    {
      ref opAssign(spec : type)(in spec[] b)
      {
        static if (r == 1)
        {
          data[0] = to!(type[])(b);

          return this;
        }
        else
        {
          static if (r == DYNAMIC)
          {
            data.length = b.length;
          }

          foreach (i, val; b)
          {
            data[i] = [val];
          }

          return this;
        }
      }
    }

    // Op Assignment Operator Overloading {{{1
    ref opOpAssign(string op, spec : type)(in auto ref spec b) if (op == "*" || op == "/")
    {
      return this = opBinary!op(b);
    }

    ref opOpAssign(string op, spec : type, size_t br, size_t bc)(in auto ref Matrix!(spec, br, bc) b) if (op == "+" || op == "-" || op == "*")
    {
      return this = opBinary!op(b);
    }

    // Index Operator Overloading {{{1
    static if (r == 1 || c == 1)
    {
      // getter
      mixin("const " ~ op_index_one);

      // setter
      mixin(op_index_one);
    }
    else
    {
      // getter
      mixin("const " ~ op_index_two);

      // setter
      mixin(op_index_two);
    }
    // }}}
  }

  pure nothrow const
  {
    // Unary Operator Overloading {{{1
    auto opUnary(string op: "-")()
    {
      return opBinary!"*"(-1);
    }

    // Binary Operator Overloading {{{1
    auto opBinary(string op, spec)(in auto ref spec b) if (isNumeric!spec && (op == "*" || op == "/"))
    {
      alias commontype = CommonType!(type, spec);

      Matrix!(commontype, r, c) that = data;

      foreach (ref dat; that.data)
      {
        mixin("dat[] " ~ op ~ "= b;");
      }

      return that;
    }

    // На выходе: тип строк - max(r, br), тип столбцов - max(c, bc).
    auto opBinary(string op, spec, size_t br, size_t bc)(in auto ref Matrix!(spec, br, bc) b) if (isNumeric!spec && (op == "+" || op == "-"))
    {
      assert(size == b.size, "size != b.size.");

      alias commontype = CommonType!(type, spec);

      // Фиксированность в приоритете
      Matrix!(commontype, max(r, br), max(c, bc)) that = b.data;

      foreach (i, ref dat; that.data)
      {
        foreach (j, ref elem; dat)
        {
          mixin("elem " ~ op ~ "= data[i][j];");
        }
      }

      return that;
    }

    // Multiplication
    // На выходе: тип строк - this.r, тип столбцов - b.c.
    auto opBinary(string op: "*", spec, size_t br, size_t bc)(in auto ref Matrix!(spec, br, bc) b) if (isNumeric!spec)
    {
      assert(cols == b.rows, "cols != b.rows.");

      alias commontype = CommonType!(type, spec);

      Matrix!(commontype, r, bc) that = new commontype[][](rows, b.cols);

      // Для удобного перемножения
      auto c = b.transpose;

      foreach (i, const ref row_a; data)
      {
        foreach (j, const ref row_c; c.data)
        {
          that.data[i][j] = dotProduct(row_a, row_c);
        }
      }

      return that;
    }

    // Binary Right Operator Overloading {{{1
    auto opBinaryRight(string op: "*", spec)(in auto ref spec b) if (isNumeric!spec)
    {
      return opBinary!op(b);
    }
    // }}}
  }

  const auto toString()
  {
    auto meta = to!(string[][])(data);
    size_t width;
    string[] result;

    foreach (const ref row; meta)
    {
      foreach (const ref element; row)
      {
        width = max(element.length, width);
      }
    }

    foreach (ref row; meta)
    {
      foreach (ref element; row)
      {
        element = rightJustify(element, width);
      }

      result ~= row.join(" ");
    }

    return result.join("\n");
  }

  pure nothrow const @safe @property
  {
    // Public Properties {{{1
    auto rows()
    {
      return data.length;
    }

    auto cols()
    {
      return data.empty ? 0 : data.front.length;
    }

    auto size()
    {
      return [rows, cols];
    }

    /++
     + Приведение матрицы к треугольному виду.
     +/
    auto triangular_view()
    {
      assert(rows > 1 && cols > 1, "Incorrect size of the matrix.");

      selftype that = data;

      // NOTE: Перебираем столбцы (для удобства заменяем строками)
      foreach (i, const ref row_i; that.data[0 .. $-1]) // necessarily that
      {
        assert(row_i[i] != 0, "First minor == 0.");

        // NOTE: Идем по строкам
        foreach (ref row_j; that.data[i+1 .. $])
        {
          row_j[] -= row_i[] * row_j[i] / row_i[i];
        }
      }

      return that;
    }

    /++
     + Приведение матрицы к диагональному виду.
     + Последовательно вычитаем текущую строку матрицы из всех верхних
     + строк, чтобы занулить их вторые элементы.
     +/
    auto diagonal_view()
    {
      auto that = triangular_view;

      foreach (i; 1 .. rows)
      {
        const auto row_i = that.data[i];

        foreach (ref row_j; that.data[0 .. i])
        {
          row_j[] -= row_i[] * row_j[i] / row_i[i];
        }
      }

      return that;
    }

    /++
     + Определитель матрицы.
     + Метод Гаусса (приведение матрицы к треугольному виду).
     +/
    auto determinant()
    {
      assert(rows == cols, "Non-square matrix.");

      auto that = triangular_view;
      type result = rows == 0 ? 0 : 1;

      foreach (i; 0 .. rows)
      {
        result *= that.data[i][i];
      }

      return result;
    }

    auto transpose()
    {
      Matrix!(type, c, r) that = new type[][](cols, rows);

      foreach (i; 0 .. rows)
      {
        foreach (j; 0 .. cols)
        {
          that.data[j][i] = data[i][j];
        }
      }

      return that;
    }

    /++
     + Обращение матрицы.
     + Допишем справа от A единичную матрицу.
     + Приведём полученную матрицу к диагональному виду, а потом приведём к
     + единичной матрице (диагональной и единичной будет только левая половина
     + матрицы). В результате правая половина матрицы будет равна
     + обратной матрице матрицы A.
     +/
    auto inverse()
    {
      assert(determinant != 0, "Determinant == 0.");

      // Only dynamic (concatenation rows).
      Matrix!type that = data;
      auto identity = Matrix!type.identity(rows, cols);

      foreach (i, ref row; that.data)
      {
        row ~= identity.data[i];
      }

      that.do_diagonal_view();

      foreach (i, ref row; that.data)
      {
        row[] /= row[i];
        row = row[$/2 .. $];
      }

      return selftype(that.data); // cpctor is not nothrow
    }
    // }}}
  }

  pure nothrow const static @safe
  {
    // Static Methods {{{1
    static if (r == DYNAMIC && c == DYNAMIC)
    {
      auto constant(spec : type)(in size_t rows, size_t cols, in auto ref spec val)
      {
        auto that = selftype(rows, cols);

        that.set_constant(val);

        return that;
      }

      auto zero(in size_t rows, in size_t cols)
      {
        return constant(rows, cols, 0);
      }

      auto ones(in size_t rows, in size_t cols)
      {
        return constant(rows, cols, 1);
      }

      auto diagonal(spec : type)(in size_t rows, in size_t cols, in auto ref spec val)
      {
        auto that = selftype(rows, cols);

        that.set_diagonal(val);

        return that;
      }

      auto identity(in size_t rows, in size_t cols)
      {
        return diagonal(rows, cols, 1);
      }
    }
    else static if (r == DYNAMIC || c == DYNAMIC)
    {
      auto constant(spec : type)(in size_t len, in auto ref spec val)
      {
        auto that = selftype(len);

        that.set_constant(val);

        return that;
      }

      auto zero(in size_t len)
      {
        return constant(len, 0);
      }

      auto ones(in size_t len)
      {
        return constant(len, 1);
      }

      auto diagonal(spec : type)(in size_t len, in auto ref spec val)
      {
        auto that = selftype(len);

        that.set_diagonal(val);

        return that;
      }

      auto identity(in size_t len)
      {
        return diagonal(len, 1);
      }
    }
    else
    {
      auto constant(spec : type)(in auto ref spec val)
      {
        selftype that;

        that.set_constant(val);

        return that;
      }

      auto zero()
      {
        return constant(0);
      }

      auto ones()
      {
        return constant(1);
      }

      auto diagonal(spec : type)(in auto ref spec val)
      {
        selftype that;

        that.set_diagonal(val);

        return that;
      }

      auto identity()
      {
        return diagonal(1);
      }
    }
    // }}}
  }
}

pure nothrow @safe unittest
{
  import std.meta: AliasSeq;

  alias Types = AliasSeq!(int, float, double, real);

  // Constructors {{{1
  {
    foreach (T; Types)
    {
      Matrix!(T, 3, 3) m1;
      Matrix!(T, DYNAMIC, 3) m2;
      Matrix!(T, 3, DYNAMIC) m3;
      Matrix!(T, DYNAMIC, DYNAMIC) m4;
    }
  }

  {
    Matrix3i m1;

    // Copy constructor (only identically types).
    auto m2 = m1;

    m1.data[0][0] = 1;

    assert(m1.data[0][0] != m2.data[0][0]);

    Matrix3r m3 = m2;

    m3.data[0][0] = 1;

    assert(m3.data[0][0] != m2.data[0][0]);
  }

  {
    foreach (T; Types)
    {
      SquareMatrix!(T, 3) m1 = new T[3][3];
      Vector!(T, 3) v1 = [0, 0, 0];
      RowVector!(T, 2) r1 = [0, 0];
    }
  }

  {
    foreach (T; Types)
    {
      auto m1 = Matrix!T(2, 2);
      auto m2 = Matrix!T(m1); // ref
      auto m3 = Matrix!T(Matrix!T()); // not ref
    }
  }

  // Static Methods {{{1
  {
    foreach (T; Types)
    {
      auto m1 = Matrix!T.constant(2,2,2);
      auto m2 = Matrix!T.zero(2,2);
      auto m3 = Matrix!T.ones(2,2);
      auto m4 = Matrix!T.diagonal(2,2,2);
      auto m5 = Matrix!T.identity(2,2);
      auto m6 = Matrix!(T, DYNAMIC, 2).constant(3,3);
      auto m7 = Matrix!(T, 2, DYNAMIC).constant(3,3);
      auto m8 = Matrix!(T, 2, 2).constant(3);
    }
  }

  // Public Methods {{{1
  {
    MatrixXi m1;
    Matrix!(float, DYNAMIC, 3) m2;
    Matrix!(real, 3, DYNAMIC) m3;
    Matrix3i m4;

    m1.set_constant(3);
    m2.set_zero();
    m2.set_ones();
    m3.set_diagonal(1);
    m4.set_identity();

    m1.init(2,2);
    m2.init(2);
    m3.init(4);
  }

  {
    auto m1 = MatrixXi(2,2);
    auto m2 = m1.set_constant(4).set_constant(4);
    auto m3 = m1.set_zero().set_zero();
    auto m4 = m1.set_ones().set_ones();
    auto m5 = m1.set_diagonal(7).set_diagonal(7);
    auto m6 = m1.set_identity().set_identity();
  }

  {
    auto m1 = MatrixXi();
    auto m2 = m1.init(2,2).init(2,2);
    auto m3 = m1.init_constant(2,2,4).init_constant(2,2,4);
    auto m4 = m1.init_zero(2,2).init_zero(2,2);
    auto m5 = m1.init_ones(2,2).init_ones(2,2);
    auto m6 = m1.init_diagonal(2,2,7).init_diagonal(2,2,7);
    auto m7 = m1.init_identity(2,2).init_identity(2,2);
  }

  // Properties {{{1
  {
    Matrix4d m1 = [[3,2,3,4],[4,4,3,2],[1,4,4,3],[2,3,1,1]];
    Matrix!(float, DYNAMIC, 4) m2 = [[3,2,3,4],[4,4,3,2],[1,4,4,3],[2,3,1,1]];
    Matrix!(float, 4, DYNAMIC) m3 = [[3,2,3,4],[4,4,3,2],[1,4,4,3],[2,3,1,1]];
    MatrixXd m4 = [[3,2,3,4],[4,4,3,2],[1,4,4,3],[2,3,1,1]];

    auto tv1 = m1.triangular_view;
    auto tv2 = m2.triangular_view;
    auto tv3 = m3.triangular_view;
    auto tv4 = m4.triangular_view;

    auto dv1 = m1.diagonal_view;
    auto dv2 = m2.diagonal_view;
    auto dv3 = m3.diagonal_view;
    auto dv4 = m4.diagonal_view;

    auto d1 = m1.determinant;
    auto d2 = m2.determinant;
    auto d3 = m3.determinant;
    auto d4 = m4.determinant;

    auto t1 = m1.transpose;
    auto t2 = m2.transpose;
    auto t3 = m3.transpose;
    auto t4 = m4.transpose;

    auto i1 = m1.inverse;
    auto i2 = m2.inverse;
    auto i3 = m3.inverse;
    auto i4 = m4.inverse;
  }

  // Unary Operator Overloading {{{1
  {
    auto m1 = -Matrix2i();
  }
  // }}}
}

pure nothrow unittest
{
  // Binary Operator Overloading {{{1
  {
    Matrix2i m1;
    auto k1 = 3.0L;

    auto m2 = m1 / k1; // ref
    auto m3 = k1 * m1; // right ref
    auto m4 = m1 / 3.0L; // not ref
    auto m5 = 3.0L * m1; // right not ref
  }

  {
    Matrix2i m1;
    Matrix2i m2;

    auto m3 = m1 + m2; // ref
    auto m4 = m1 - Matrix2i(); // not ref
  }

  {
    Matrix2i m1;
    Matrix2i m2;

    auto m3 = m1 * m2; // ref
    auto m4 = m1 * Matrix2i(); // not ref
  }

  {
    auto m1 = MatrixXr.constant(2,2,3);
    auto m2 = MatrixXr.zero(2,3);
    auto m3 = MatrixXr.ones(2,3);
    auto m4 = MatrixXr.diagonal(2,3,2);
    auto m5 = MatrixXr.identity(2,2);

    m1 += MatrixXr.constant(2,2,3);
    m2 -= MatrixXr.zero(2,3);
    m3 *= MatrixXr.diagonal(3,2,2);
    m3 *= MatrixXr.identity(2,2);
  }

  // Assignment Operator Overloading {{{1
  {
    Matrix3i m1;
    MatrixXi m2;
    Matrix!(int, DYNAMIC, 2) m3;
    Matrix!(int, 2, DYNAMIC) m4;
    Matrix!(int, DYNAMIC, 2) m5;

    m1 = new int[3][3];
    m2 = [[2,2],[2,2]];

    m3 = m2;
    m3 = [[2,2]];
    m4 = [[2],[2]];

    m2 = m1;
    m2 = m3;
    m2 = m4;

    m3 = [[2,2],[2,2]];
    m4 = [[2,2],[2,2]];

    m3 = m5;
    m3 = m4;

    Matrix2f m6;
    Matrix!(int, DYNAMIC, 2) m7;
    Matrix!(int, 2, DYNAMIC) m8;

    m7 = [[2,2],[2,2]];
    m8 = [[2,2],[2,2]];
    m2 = [[2,2],[2,2]];

    m6 = m7;
    m6 = m8;
    m6 = m2;

    m2 = m1 + m1; // not ref
  }

  {
    VectorXf v1;
    RowVectorXf v2;

    v1 = [1,2];
    v2 = [1,2];
  }

  // Op Assignment Operator Overloading {{{1
  {
    MatrixXi m1;
    MatrixXf m2;

    m1 *= 3;
    m1 /= 3;
    m2 += m1;
    m2 -= m1;
    m2 *= m1;
  }
  // }}}
}

pure nothrow @safe unittest
{
  // Index Operator Overloading {{{1
  {
    Matrix3i m1;
    const Matrix3i m2;

    m1[1,1] = m2[1,1];
  }
  // }}}
}
