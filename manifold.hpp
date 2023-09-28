#pragma once

#include <iostream>
#include <cmath>

#include "defs.hpp"

namespace QME 
{

class Manifold {
public:
    /* Abstract class */
    virtual ~Manifold(void) = 0;
    virtual Variable Retract(const Vector& x, const Vector& v) const = 0;
    virtual double Norm(const Vector& x, const Vector& v) const = 0;
    virtual void Gradient(const Vector& x, const Vector& y, Vector& grad) const = 0;
    virtual Vector Projection(const Variable x, const Vector v) const = 0;
    virtual Vector RandomManifold(void) const = 0;

    /*Compute the Riemannian gradient from the Euclidean gradient of a function;
		The function is defined in "prob".
		egf is the Euclidean gradient; the output is the Riemannian gradient.
		It is a pure virtual function. It must be overloaded by derived class */
    virtual Vector EucGradToGrad(const Variable& x, const Vector& egf) const = 0;
    
    /*Compute the Riemannian action of Hessian from the Euclidean action of Hessian of a function;
		It is a pure virtual function.It must be overloaded by derived class */
    virtual Vector EucHvToHv(const Variable& x, const Vector& egrad, const Vector& ehess, const Vector& x_dot) const = 0;
};

} // end of namespace QME