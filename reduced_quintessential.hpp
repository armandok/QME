#pragma once

#include <iostream>
#include <cmath>

#include "manifold.hpp"

namespace QME 
{

class ReducedQuintessential : public Manifold {
public:
	ReducedQuintessential();
	ReducedQuintessential(int r);
	virtual ~ReducedQuintessential(void);
	virtual Variable Retract(const Vector& x, const Vector& v) const override;
	virtual double Norm(const Vector& x, const Vector& v) const override;
	virtual void Gradient(const Vector& x, const Vector& y, Vector& grad) const override;
	virtual Vector Projection(const Variable x, const Vector v) const override;
	virtual Vector RandomManifold(void) const override;

	/*Compute the Riemannian gradient from the Euclidean gradient of a function;
	egf is the Euclidean gradient; the output is the Riemannian gradient.
	*/
	virtual Vector EucGradToGrad(const Variable& x, const Vector& egf) const override;

	/*Compute the Riemannian action of Hessian from the Euclidean action of Hessian of a function;
	*/
	virtual Vector EucHvToHv(const Variable& x, const Vector& egrad, const Vector& ehess, const Vector& x_dot) const override;

	void set_rank(size_t rank) {m_rank = rank;};

	Vector GetConstraintsVal(const Variable x) const;

	Vector GetGradConstraintsVal(const Variable x, const Variable g) const;

	Variable RetractHeuristic(const Vector U) const;
	Variable RetractNewton(const Vector X) const;
	Vector GetConstraintsJacobian(const Vector X) const;

	/** Converts a matrix on the manifold with the bm arrangement to the 
     * stiefel arrangement */
    Variable to_stiefel(const Variable &Y) const;

    /** Converts a matrix on the manifold with the stiefel arrangement to the 
     * bm arrangement */
    Variable to_bm(const Variable &V) const;

private:
	int m_rank;
};


} // end of namespace QME