from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User
from django.core import serializers
import json
import string
import random
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import *
from boto.mturk.price import Price
import object_permissions

from images.models import ImageModel
from tags.models import Part,Category,CategoryTag,Annotation,Entity,PartTag,bounding_box
from tags.views import save_annos
from util import *
import math

# An annotation task defines a set of images that are to be annotated according to a particular
# annotation model.  Currently, there is a one-to-one correspondence between an AnnotationTask
# and a HitType.  The other classes in this file (other than AnnotationTask) are wrappers around
# different types of constructs on Mechanical Turk.  It is intended for these to be defined
# in a generic way, that is independent of our particular notion of image annotation models.
# We have decoupled the classes AnnotationTask and HitType because maybe in the future we want
# to have AnnotationTasks that could be collected using some resource other than MTurk (e.g. 
# ODesk)
class AnnotationTask(models.Model):
    hit_type = models.ForeignKey('HitType', null=True, default=None)
    
    # Model defining the type of annotation to collect
    anno_model = models.ForeignKey(Part, null=True, default=None)
    
    # Optional additional string defining the type of annotation to collect
    anno_type = models.CharField(max_length=20, db_index=True, null=True)
   
    # Optional string describing the way of breaking down the annotation task into a series
    # of simpler tasks.  Selecting "everything" collects objects and parts in a single HIT.  
    # Selecting 'objects' just collects the object region (no parts). Selecting 'all parts' assumes
    # a previous annotation task has collected object regions and collects all part labels
    # for each object.  Selecting 'per part' also assumes object regions already exist, but this
    # time it collects each part separately in different HITs, for multiple images at a time.
    decomp_type = models.CharField(max_length=20, db_index=True, null=True, default="everything")
    
    # The maximum number of objects per image
    max_objects = models.PositiveIntegerField(default = 30)
    
    # The minimum number of objects per image
    min_objects = models.PositiveIntegerField(default = 0)
    
    # Images that are to be annotated in this task should be tagged with this particular category
    category = models.ForeignKey(Category, null=True, default=None)
    
    # Number of images a Worker should annotate in each Hit
    images_per_hit = models.PositiveIntegerField(default = 1)

    # JSON encoded data that could be used to store customizable parameters for this annotation task type
    params = models.TextField(null=True, default=None)
    
    # The user that owns this Task
    user = models.ForeignKey(User, db_index=True, null=True, default=None)
    
    class Meta:
        permissions = (
            ('view_annotationtask', 'View annotation task'),
        )
    
    def assumes_pre_existing_object(self):
      return self.decomp_type == 'all parts' or self.decomp_type == 'per part'
    
    def get_image_set(self):
        if not self.category is None:
            if self.assumes_pre_existing_object():
                images = ImageModel.objects.filter(categorytag__category = self.category, annotation__source = "MTurkConsensus")
            else:
                images = ImageModel.objects.filter(categorytag__category = self.category)
        else:
            if self.assumes_pre_existing_object():
                images = ImageModel.objects.filter(taxon=self.anno_model.id, annotation__source = "MTurkConsensus")
            else:
                images = ImageModel.objects.filter(taxon=self.anno_model.id)
        return images
        
    def get_cost(self, max_assignments):
        return math.ceil(len(self.get_image_set().all())/float(self.images_per_hit))*float(self.hit_type.reward)*float(max_assignments)

    # Build and register a set of MTurk HITs that cover all relevant images.  Assumes self.hit_type
    # has already been created and registered.  This function divides an annotation task into
    # a set of HITs (each of which defines an annotation session on MTurk)
    def build_hits(self, host, max_assignments, lifetime, frame_height):
        images = self.get_image_set()
        if self.decomp_type == 'objects':
            params = [{"parts":[self.anno_model.id]}]
        elif self.decomp_type == 'per part':
            params = []
            for p in Part.objects.filter(parent=self.anno_model):
                params.append({"parts":[p.id]})
        else:
            part_ids = []
            for p in Part.objects.filter(parent=self.anno_model):
                part_ids.append(p.id)
            params = [{"parts":part_ids}]
        for p in params:
            hit = None
            obj_view = None
            image_objs = []
            image_obj_bboxes = []
            for i in images: 
                if self.assumes_pre_existing_object():
                    objs = []
                    bboxes = []
                    tags = PartTag.objects.filter(view__part=self.anno_model, entity__annotation__source="MTurkConsensus", entity__annotation__image=i)
                    for o in tags:
                        if obj_view is None: 
                            obj_view = json.loads(serializers.serialize('json', [o.view]))[0]
                        objs.append(o)
                        bbox = bounding_box(o.view.points(o))
                        bboxes.append([bbox.x,bbox.y,bbox.width,bbox.height])
                    image_objs.append(json.loads(serializers.serialize('json', objs)))
                    image_obj_bboxes.append(bboxes)
                    p["image_objs"] = image_objs
                    p["image_obj_bboxes"] = image_obj_bboxes
                    p["obj_view"] = obj_view
                
                if not hit:
                    at = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10))
                    hit = Hit(hit_type=self.hit_type, max_assignments=max_assignments, 
                              lifetime=lifetime, frame_height=frame_height, status='Unassigned',
                              access_token=at, user=self.user)
                    hit.save()
                hit.images.add(i)
                if hit.images.count() >= self.images_per_hit:
                    hit.params = json.dumps(p)
                    hit.save()
                    hit = None
                    obj_view = None
                    image_objs = []
                    image_obj_bboxes = []
            if not hit is None:
                hit.params = json.dumps(p)
                hit.save()
        
        for h in self.hit_type.hit_set.all():
            if not h.register(host):
                return False
        return True

    # Download completed assignments from MTurk
    def get_completed_assignments(self):
        for h in self.hit_type.hit_set.all():
            h.get_completed_assignments()

    # Find assignments that have been completed and downloaded from MTurk (such that we have 
    # completed creation of a Worker instance for it).  This call should follow
    # get_completed_assignments().  It parses the completed answer of each 
    # assignment and store them to the database 
    def save_completed_assignments(self):
        assignments = Assignment.objects.filter(hit__hit_type__anno_task=self, 
                                                saved=False, worker__isnull=False)
        for a in assignments:
            save_annos(jsonData=json.loads(a.answer), assignment=a, anno_task=self)
            a.saved = True
            a.save()
    
    # Combine annotations from multiple workers into a single annotation, and score each assignment
    # by annotation quality (e.g., by comparing to other annotators or ground truth)
    # Only process HITs where at least min_completed_assignments assignments have been completed.
    # In general, an approximate threshold for whether or not two annotated objects in different
    # assignments match is if fractionCorrect of parts match
    def crowdsource_results(self, fractionCorrect=.4):
        for hit in self.hit_type.hit_set.all():
            # For each HIT that has all assignments completed and hasn't been processed yet,
            # solve for a consensus annotation for each image and score each mturk assignment
            num = hit.max_assignments #if min_completed_assignments is None else min_completed_assignments
            assignments = hit.assignment_set
            params = json.loads(hit.params) if not hit.params is None else {}
            if assignments.count() >= num:
                if assignments.all()[0].score is None:
                    assignmentScores = []
                    for a in assignments.all():
                        assignmentScores.append(0)
                    ind = 0
                    for img in hit.images.all():
                        # If a ground truth label exists, we'll use it to score each assignment and we won't
                        # generate an MTurk consensus annotation
                        gts = Annotation.objects.filter(source="User", image=img, part=self.anno_model) 
                        gt = gts.all()[0] if gts.count() > 0 else None
                        image_objs = params["image_objs"][ind] if "image_objs" in params else None
                        
                        # Generate and save an MTurk consensus annotation if no ground truth label exists
                        # Match each MTurk annotated object to ground truth or consenus objects
                        [scores,annotation] = self.match_and_score_annotations(assignments, fractionCorrect*assignments.count(), 
                                                                               img, gt=gt, image_objs=image_objs)
                        for i in range(0,assignments.count()):  # The score per image will be a number between 0 and 1
                            assignmentScores[i] += scores[i]
                        ind += 1
                    for i in range(0,assignments.count()):
                        a = assignments.all()[i]
                        a.score = assignmentScores[i]/hit.images.count() # The score will be a number between 0 and 1
                        a.save()  
    
    # Approve all assignments where the score (obtained using crowdsourcing techniques) is above 
    # min_score_to_approve.  If min_score_to_approve is None, approve all pending Assignments
    def approve_assignments(self, min_score_to_approve=None, feedback=None):
        if min_score_to_approve is None:
            assignments = Assignment.objects.filter(hit__hit_type__anno_task=self, worker__isnull=False, 
                                                    status='Created')
        else:
            assignments = Assignment.objects.filter(hit__hit_type__anno_task=self, worker__isnull=False, 
                                                    status='Created', score__gte=min_score_to_approve)
        for a in assignments:
            a.approve(feedback)
    
    # Reject all assignments where the score (obtained using crowdsourcing techniques) is below 
    # min_score_to_reject.  If min_score_to_reject is None, reject all pending Assignments
    def reject_assignments(self,  min_score_to_reject=None, feedback=None):
        if min_score_to_reject is None:
            assignments = Assignment.objects.filter(hit__hit_type__anno_task=self, worker__isnull=False, 
                                                    status='Created')
        else:
            assignments = Assignment.objects.filter(hit__hit_type__anno_task=self, worker__isnull=False, 
                                                    status='Created', score__lt=min_score_to_reject)
        for a in assignments:
            a.reject(feedback)
    
    # Return a hash of tuples of all workers who have completed assignments for this annotation task,
    # where for each tuple the first element is the Worker object, the second is the number of
    # assignments they have completed, and the 3rd is their average score
    def worker_stats(self):
        assignments = Assignment.objects.filter(hit__hit_type__anno_task=self, worker__isnull=False)
        workers = {}
        for a in assignments:
            w = a.worker
            if not str(w.id) in workers:
                workers[str(w.id)]=[w,0,0]
            workers[str(w.id)][1] += 1  # number of assignments completed by this worker
            workers[str(w.id)][2] += a.score  # sum score of assignments completed by this worker
        for w in workers:
            workers[w][2] /= workers[w][1]
        return workers

    # Return a list of all workers who have completed assignments for this annotation task
    def workers(self):
        stats = self.worker_stats()
        workers = []
        for w in stats:
            workers.append(stats[w][0])
        return workers
        

    # Build qualification types that can be used to filter workers based on the results to this 
    # annotation task.  We create 2 qualification types, one that records the average score
    # of each worker over each completed assignment, and a second that records the number of completed 
    # assignments.  It is then intended that when one creates a new annotation task, they could
    # add qualification requirements that workers must have an average score greater than T, while
    # having completed at least n assignments
    def assign_qualifications_by_score(self):
        qual_score = Qualification(name='Score on '+self.hit_type.title, auto_granted_value=0,
                                   account = self.hit_type.account, sandbox=self.hit_type.sandbox)
        qual_num   = Qualification(name='Number completed of '+self.hit_type.title, auto_granted_value=0, 
                                   account = self.hit_type.account, sandbox=self.hit_type.sandbox)
        qual_score.register()
        qual_num.register()
        stats = self.worker_stats()
        for w in stats:
            stats[w][0].assign_qualification(qual_score, value=int(stats[w][2]*1000), send_notification=False)
            stats[w][0].assign_qualification(qual_num,   value=stats[w][1], send_notification=False)
    
    # Block all workers that have completed at least min_completed assignments while obtaining an
    # average score worse than bad_score
    def block_workers(self, bad_score, min_completed=0, reason=None):
        stats = self.worker_stats()
        for w in stats:
            if (not stats[w][0].is_blocked) and stats[w][2] < bad_score and stats[w][1] >= min_completed:
                stats[w][0].block(reason)
    
    # Unblock all workers that have completed at least min_completed assignments while obtaining an
    # average score at least as good as good_enough_score
    def unblock_workers(self, good_enough_score, min_completed=0, reason=None):
        stats = self.worker_stats()
        for w in stats:
            if stats[w][0].is_blocked and stats[w][2] >= good_enough_score and stats[w][1] >= min_completed:
                stats[w][0].unblock(reason)
                
    # Grant a monetary bonus to all workers that have completed at least min_completed assignments
    # while obtaining an average score at least as good as good_score
    def grant_bonus(self, price, good_score, min_completed=0, reason=None):
        stats = self.worker_stats()
        for w in stats:
            if stats[w][2] >= good_score and stats[w][1] >= min_completed:
                stats[w][0].grant_bonus(reason=reason, price=price)

    # Disable all HITs from MTurk.  This approves any submitted assignments pending approval or rejection,
    # and disposes of the HIT and all assignment data. 
    def disable(self):
        for h in self.hit_type.hit_set.all():
            h.disable()
    
    # Dispose all HITs from MTurk that are in the Reviewable state, with all of their submitted assignments 
    # already either approved or rejected. 
    def dispose(self):
        for h in self.hit_type.hit_set.all():
            h.dispose()
    
    # Expire all HITs on MTurk, such that they are no longer available
    def expire(self):
        for h in self.hit_type.hit_set.all():
            h.expire()
    
    # If expiration_increment, extend the expiration time of a hit by expiration_increment seconds.
    # If assignments_increment, add additional assignments per HIT
    def extend(self, assignments_increment=None, expiration_increment=None):
        for h in self.hit_type.hit_set.all():
            h.extend(assignments_increment, expiration_increment)
    
    # Send an email message to all workers that have completed at least min_completed.  If
    # bad_score or good_score are specified, only send a message to workers that have score
    # worse than bad_score or better than good_score
    def notify_workers(self, subject, message_text, bad_score=None, good_score=None, min_completed=0):
        workers = []
        stats = self.worker_stats()
        for w in stats:
            if ((not good_score is None) and stats[w][2] >= good_score and stats[w][1] >= min_completed) or \
                    ((not bad_score is None) and stats[w][2] < bad_score and stats[w][1] >= min_completed) or \
                    ((good_score is None) and (bad_score is None)):
                workers.append(stats[w][0])
        return self.hit_type.notify_workers(workers=workers, subject=subject, message_text=message_text)
    
    # Switch all applicable hits from 'Reviewable' to 'Reviewing' state (or vice versa if revert is specified)
    def set_reviewing(self, revert=False):
        for h in self.hit_type.hit_set.all():
            h.set_reviewing(revert)
    
    
    # Helper function to crowdsource_results()
    # If there are multiple labeled objects per annotation, they might be in different order
    # in each assignment, and there might be a different number of labeled objects per image
    # in different assignments.  We need to compute a correspondence between objects
    # in different assignment.  We use a greedy facility location algorithm (Jain et al. STOC '02), 
    # where we first compute pairwise distances between each object in different assignments:
    def match_and_score_annotations(self, assignments, openCost, img, gt=None, image_objs=None):
        all_objects = [] # A list of 5-tuples of all annotated objects: (assignment_ind,entity_ind,isFacility,isCity,openCost)
        costs = [] # A list of pairwise cost 2-tuples between objects: (cost,all_objects_ind1,all_objects_ind2)
        objs = {} # For each assignment, a list of all objects 
        openCosts = {} 
        cityDisallowedCityNeighbors = {}  # for each city, a list of cities that aren't allowed to be connected to the same facility
        preexisting = self.assumes_pre_existing_object()   # If true, we know all objects were annotated in the same order
        
        # Each annotated object is a city, meaning it will need to be matched to a prototypical object (facility).
        # If there is no ground truth annotation, each annotated object is a candidate facility that can be
        # opened at cost openCost.  openCost should typically be set to something like the number of assignments 
        # that must agree on the presence of an object for it to be considered to be a consensus object (e.g. half
        # the number of assignments)
        for i in range(0,assignments.count()):
            objs[i] = assignments.all()[i].annotation_set.filter(image=img)[0].entity_set.all()
            for j in range(0,objs[i].count()):
                all_objects.append((i,j,gt is None,True,openCost))
        
        if not gt is None:
            # If a ground truth label is present, each ground truth object is a facility that can be
            # opened at no cost, such that each annotated object in an mturk assignment must be matched to a
            # ground truth object
            objs[-1] = gt.entity_set.all()
            for j in range(0,objs[-1].count()):
                all_objects.append((-1,j,True,False,0))
        
        # A dummy facility is intended to handle object labels that don't get matched to a ground truth / consensus
        # object. It can be opened at no cost, but the cost of matching it to annotated objects will be high
        dummy = Entity(annotation=Annotation(), name='__dummy__')
        objs[-2] = [dummy]
        all_objects.append((-2,0,True,False,0))
        
        # Compute matching costs between all facility-city pairs
        for i in range(0,len(all_objects)):
            cityDisallowedCityNeighbors[i] = []
            if all_objects[i][2]: openCosts[i] = all_objects[i][4]
            o1 = objs[all_objects[i][0]][all_objects[i][1]]
            for j in range(0,len(all_objects)):
                if all_objects[j][3]: # object j is an allowable city
                    # An object cannot be matched to another object in the same assignment; however, if 
                    # objects are not pre-existing, it can be matched to any object in any different assignment.
                    # If objects are pre-existing, it can only be matched to the same object index in other
                    # assignments.  It can always be matched to itself at (presumably) 0 cost.
                    if all_objects[i][2] and \
                            ((all_objects[i][0] != all_objects[j][0] and \
                                  ((not preexisting) or all_objects[i][1] != all_objects[j][1])) or \
                                 (all_objects[i][0] == all_objects[j][0] and all_objects[i][1] == all_objects[j][1])):
                        o2 = objs[all_objects[j][0]][all_objects[j][1]]
                        c = o1.matching_cost(o2)
                        costs.append((c,i,j))
                    # Different labeled objects in the same assignment cannot be assigned to the same ground 
                    # truth object
                    elif all_objects[i][0] == all_objects[j][0] and all_objects[i][1] != all_objects[j][1]:
                        cityDisallowedCityNeighbors[i].append(j) 

        # Now choose a set of facilities, where a facility is one object in a particular assignment (or ground truth).
        # Each facility is considered to be the best or most prototypical example of an object, and
        # instances of the same object in other assignments should be connected to it.  A facility 
        # location algorithm optimizes the choice of facilities and connections to it that minimizes 
        # the sum matching costs plus the sum cost of opening each facility.  Thus it effectively 
        # chooses how many objects are in an image, a prototypical annotation of that object, and a 
        # matching between that object and objects in other assignments 
        [facilities,total_cost] = FacilityLocation(cityFacilityCosts=costs).solve(openFacilityCosts=openCosts)
        #import pdb; pdb.set_trace()
        
        if gt is None:
            # Save the consensus annotation to the database
            if image_objs is None:
                annotation = Annotation(part=self.anno_model, source="MTurkConsensus", image=img, anno_task=self)
                annotation.save()
                for o in facilities:
                    i = all_objects[o][0]
                    j = all_objects[o][1]
                    if objs[i][j] != dummy:   
                        # for all facilities that are not the dummy facility, create and save an object
                        e = objs[i][j].save_copy(annotation=annotation)
            else:
                for o in facilities:
                    i = all_objects[o][0]
                    j = all_objects[o][1]
                    if objs[i][j] != dummy:
                        e = objs[i][j].save_copy(entity=image_objs[j].entity)
        else:
            annotation = gt

        # Score each assignment by agreement to the ground truth or consensus annotation
        facilityMissingCosts={}
        numMatches={}
        assignments_costs = []
        assignments_scores = []
        for a in range(0,assignments.count()):
            assignments_costs.append(0)
            numMatches[a] = {}
            for o in facilities:
                numMatches[a][o] = 0
        for o in facilities:
            f = objs[all_objects[o][0]][all_objects[o][1]]
            facilityMissingCosts[o] = f.matching_cost(dummy) if f != dummy else 0
            for i in facilities[o]:
                # cost of matching an annotated object to a ground truth object
                a = all_objects[i][0]
                assignments_costs[a] += facilities[o][i]  
                numMatches[a][o] += 1
        for a in range(0,assignments.count()):
            for o in numMatches[a]: 
                if numMatches[a][o] == 0:
                    # cost of leaving a ground truth object (excluding the dummy facility) unmatched
                    assignments_costs[a] += facilityMissingCosts[o]
                elif numMatches[a][o] > 1:
                    # cost of matching multiple objects to the same facility
                    assignments_costs[a] += facilityMissingCosts[o]*(numMatches[a][o]-1)
            assignments_scores.append(max(0, 1 - assignments_costs[a] / max(1,annotation.entity_set.count())))
        
        return (assignments_scores,annotation)
    
object_permissions.register(['read', 'write', 'remove', 'manage'], AnnotationTask)

# Defines an account on Amazon (aws), which will be used to pay for MTurk HITs.  See
# http://blogs.aws.amazon.com/security/post/Tx1R9KDN9ISZ0HF/Where-s-my-secret-access-key
# for info on where to find access keys and secret access keys
class Account(models.Model): 
    # A string description of this account
    name = models.CharField(max_length=100, db_index=True, unique=True, null=True)
    
    # Requester's Access Key ID for MTurk API, see "AWSAccessKeyId" in MTurk documentation
    aws_access_key = models.CharField(max_length=100, null=True)
    
    # Secret Access Key for MTurk API
    aws_secret_access_key = models.CharField(max_length=100, null=True)
    
    # The user that owns this Account
    user = models.ForeignKey(User, db_index=True, null=True, default=None)
    
    def connection(self, sandbox):
        h = 'mechanicalturk.sandbox.amazonaws.com' if sandbox else 'mechanicalturk.amazonaws.com'
        return MTurkConnection(aws_access_key_id=self.aws_access_key, 
                               aws_secret_access_key=self.aws_secret_access_key, host=h)
    
    def get_balance(self):
        mtc = self.connection(self, False)
        rs = mtc.get_account_balance()
        if not rs.status:
            return False
        return rs[0]

object_permissions.register(['read', 'write', 'remove'], Account)


# A Set of HITs with the same HIT type will be grouped together on MTurk.  A worker can choose to
# do multiple HITs of the same annotation task in a row.  In our case, each HIT will be comprised
# of annotating a set of images according to some annotation model, and should typically be
# sized to take maybe a couple of minutes.  HITs of a certain HitType will come up when a
# worker searches for HITs on MTurk, with the fields title, description, and keywords defining the
# search results and query mechanism
class HitType(models.Model):
    # The AnnotationTask associated with this HitType
    anno_task = models.ForeignKey('AnnotationTask', null=True, default=None)

    # Title of this type of annotation task (shows up on MTurk)
    title = models.CharField(max_length=100, db_index=True, null=True)
    
    # A text description of this annotation task (shows up on MTurk)
    description = models.TextField(null=True)
    
    # Keywords tagging this annotation task (shows up on MTurk)
    keywords = models.TextField(null=True)
    
    # Reward in dollars for completing a HIT
    reward = models.FloatField(default = 0.01)
    
    # Maximum duration of this HIT in seconds
    duration = models.PositiveIntegerField(default = 3600)
    
    # Time in seconds after which a completed assignment is approved automatically
    auto_approval_delay = models.PositiveIntegerField(default = 172800)
    
    # A list of qualifications a worker must meet to be able to do this HIT.  This field
    # should be a JSON formatted string containing an array of qualification requirements.
    # Each requirement must have 4 fields: "qualification_type", "comparator", "value", "required_to_preview"
    qualifications = models.TextField(null=True)
    
    # HIT Type id when interfacing with the MTurk API
    mturk_id = models.CharField(max_length=100, db_index=True, null=True)
    
    # If true, this HIT will be in the sandbox instead of officially on MTurk
    sandbox = models.BooleanField(default=True)
    
    # The MTurk account used to pay for these HITs
    account = models.ForeignKey(Account, db_index=True, null=True, default=None)

    # The user that owns this HIT type
    user = models.ForeignKey(User, db_index=True, null=True, default=None)
    
    # Register a HitType on Amazon MTurk
    def register(self):
        mtc = self.account.connection(self.sandbox)
        rs = mtc.register_hit_type(title=self.title, description=self.description, 
                                   reward=Price(self.reward), duration=self.duration, 
                                   keywords=self.keywords, 
                                   approval_delay=self.auto_approval_delay, 
                                   qual_req=self.build_qualifications())
        if rs.status:
            self.mturk_id = rs[0].HITTypeId
            self.save()
            return True
        return False
    
    def notify_workers(self, workers, subject, message_text):
        mtc = self.account.connection(self.sandbox)
        worker_ids = []
        for w in workers:
            worker_ids.append(w.mturk_id)
        rs = mtc.notify_workers(worker_ids=worker_ids, subject=subject, message_text=message_text)
        return rs.status
    
    # Helper function to convert qualification requirements (encoded as a JSON string) into the 
    # format required by the boto MTurk API
    def build_qualifications(self):
        if self.qualifications is None or self.qualifications=='':
            return None
        qs = json.loads(self.qualifications)
        qr = Qualifications()
        for q in qs:
            if q['qualification_type'] == "Adult": 
                qr.add(AdultRequirement(q['comparator'], q['value'], 
                                        q['required_to_preview']))
            elif q['qualification_type'] == "Locale": 
                qr.add(LocaleRequirement(q['comparator'], q['value'], 
                                         q['required_to_preview']))
            elif q['qualification_type'] == "NumberHitsApproved": 
                qr.add(NumberHitsApprovedRequirement(q['comparator'], q['value'], 
                                                     q['required_to_preview']))
            elif q['qualification_type'] == "PercentAssignmentsAbandoned": 
                qr.add(PercentAssignmentsAbandonedRequirement(q['comparator'], q['value'], 
                                                              q['required_to_preview']))
            elif q['qualification_type'] == "PercentAssignmentsApproved": 
                qr.add(PercentAssignmentsApprovedRequirement(q['comparator'], q['value'], 
                                                             q['required_to_preview']))
            elif q['qualification_type'] == "PercentAssignmentsRejected": 
                qr.add(PercentAssignmentsRejectedRequirement(q['comparator'], q['value'], 
                                                             q['required_to_preview']))
            elif q['qualification_type'] == "PercentAssignmentsReturned": 
                qr.add(PercentAssignmentsReturnedRequirement(q['comparator'], q['value'], 
                                                             q['required_to_preview']))
            elif q['qualification_type'] == "PercentAssignmentsSubmitted": 
                qr.add(PercentAssignmentsSubmittedRequirement(q['comparator'], q['value'], 
                                                              q['required_to_preview']))
            else: 
                qr.add(Requirement(Qualification.objects.get(pk=q['qualification_type']).mturk_id, 
                                   q['comparator'], q['value'], q['required_to_preview']))
        return qr
    
    # TODO: validate fields, qualifications, get_reviewable_hits(), approve_all_assignments()

object_permissions.register(['read', 'write', 'remove'], HitType)

# a HIT defines an annotation task for an MTurk worker.  The worker must complete the entire HIT
# and then submit his results
class Hit(models.Model):
    # JSON encoded data that could be used to store customizable parameters for this HIT
    params = models.TextField(null=True, default=None)
    
    # A list of images to be annotated
    images = models.ManyToManyField(ImageModel)
    
    # The type of HIT (groups HITs of a similar annotation task on MTurk)
    hit_type = models.ForeignKey(HitType, null=True, default=None)
    
    # assignments of this HIT to (possibly multiple) MTurk workers
    # assignment_set throught assignment.hit
    
    # maximum number of workers that can be assigned to this HIT
    max_assignments = models.PositiveIntegerField(default = 0)
    
    # maximum lifetime in seconds of this HIT
    lifetime = models.PositiveIntegerField(default = 0)
    
    frame_height = models.PositiveIntegerField(default = 800)
    
    # One of 'Unassigned', 'Reviewable', 'Disposed', 'Reviewing', 
    status = models.CharField(max_length=20, db_index=True, null=True)
    
    # The number of assignments for this HIT that have been completed so far
    num_completed_assignments = models.PositiveIntegerField(default = 0, db_index=True)
    
    # HIT id when interfacing with the MTurk API
    mturk_id = models.CharField(max_length=100, db_index=True, null=True)
    
    # This HIT is only accessible if the caller knows the access token (which is encoded in the url request).
    # The purpose is to prevent people from accessing our HITs by guessing the url
    access_token = models.CharField(max_length=100, db_index=True, null=True)
    
    # The user that owns this HIT
    user = models.ForeignKey(User, db_index=True, null=True, default=None)
    
    # url from which this HIT can be accessed
    def url(self):
        return '/mturk/hits/' + str(self.id) + "?at=" + self.access_token
        
    # Registers this HIT on Amazon Mechanical Turk
    def register(self, host):
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        rs = mtc.create_hit(hit_type=self.hit_type.mturk_id, 
                            question=ExternalQuestion(external_url="https://" + host + self.url(),
                                                      frame_height=self.frame_height), 
                            lifetime=self.lifetime, max_assignments=self.max_assignments)
        if rs.status:
            self.mturk_id = rs[0].HITId
            self.save()
            return True
        return False
    
    # Query Amazon Mechanical Turk for completed assignments, storing them to our database,
    # and creating new Worker instances in our database if necessary
    def get_completed_assignments(self):
        if self.mturk_id is None:
            return False
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        if self.num_completed_assignments < self.max_assignments and not self.status=='Disposed':
            rs = mtc.get_assignments(hit_id=self.mturk_id, page_size=self.max_assignments)
            if rs.status:
                old_completed = self.num_completed_assignments
                for ra in rs:
                    if Assignment.objects.filter(mturk_id = ra.AssignmentId).count() == 0:
                        self.num_completed_assignments += 1
                        ws = Worker.objects.filter(mturk_id=ra.WorkerId)
                        if ws.count() > 0:
                            w = ws[0]
                        else:
                            w = Worker(mturk_id=ra.WorkerId, account=self.hit_type.account, 
                                       sandbox=self.hit_type.sandbox)
                            w.save()
                        a = Assignment(hit=self, worker=w, answer=ra.answers[0][0].fields[0], status='Created', 
                                       mturk_id=ra.AssignmentId, accept_time=ra.AcceptTime, 
                                       submit_time=ra.SubmitTime, saved=False)
                        a.save()
                if old_completed != self.num_completed_assignments:
                    if self.num_completed_assignments >= self.max_assignments:
                        self.status='Reviewable'
                    self.save()
            return rs.status
        return True
    
    # TODO: validation
    def disable(self):
        if self.mturk_id is None or self.status=='Disposed':
            return False
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        rs = mtc.disable_hit(hit_id=self.mturk_id)
        if rs.status:
            self.status = 'Disposed'
            self.save()
            #self.assignment_set.delete()
        return rs.status
    
    def dispose(self):
        if self.mturk_id is None or not self.status=='Reviewable':
            return False
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        rs = mtc.dispose_hit(hit_id=self.mturk_id)
        if rs.status:
            self.status = 'Disposed'
            self.save()
        return rs.status
    
    def expire(self):
        if self.mturk_id is None:
            return False
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        rs = mtc.expire_hit(hit_id=self.mturk_id)
        return rs.status
    
    def extend(self, assignments_increment=None, expiration_increment=None):
        if self.mturk_id is None:
            return False
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        rs = mtc.extend_hit(hit_id=self.mturk_id, assignments_increment=assignments_increment, 
                            expiration_increment=expiration_increment)
        return rs.status
    
    def set_reviewing(self, revert=False):
        req_status = self.status = 'Reviewing' if revert else 'Reviewable' 
        if self.mturk_id is None or not self.status==req_status:
            return False
        mtc = self.hit_type.account.connection(self.hit_type.sandbox)
        rs = mtc.set_reviewing(hit_id=self.mturk_id, revert=revert)
        if rs.status:
            self.status = 'Reviewable' if revert else 'Reviewing' 
            self.save()
        return rs.status
        

# An MTurk worker
class Worker(models.Model):
    # Worker id when interfacing with the MTurk API
    mturk_id = models.CharField(max_length=100, db_index=True, null=True)
    
    # The MTurk account used to pay for these HITs
    account = models.ForeignKey(Account, db_index=True, null=True, default=None)

    # If true, this HIT will be in the sandbox instead of officially on MTurk
    sandbox = models.BooleanField(default=True)
    
    # If true, this worker is blocked from doing our HITs
    is_blocked = models.BooleanField(default=False)
    
    # Optionally, one can assign a score to this worker using a crowdsourcing algorithm
    score = models.FloatField(null=True, db_index=True)

    def assign_qualification(self, qualification, value=1, send_notification=True):
        if qualification.mturk_id is None or self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        rs = mtc.assign_qualification(qualification_type_id=qualification.mturk_id, worker_id=self.mturk_id, 
                                      value=value, send_notification=send_notification)
        if rs.status:
            wq = WorkerQualification(worker=self, qualification=qualification, value=value)
            wq.save()
        return rs.status
    
    def revoke_qualification(self, qualification, reason=None):
        if qualification.mturk_id is None or self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        wqs = WorkerQualification.objects.filter(worker=self, qualification=qualification)
        for wq in wqs:
            rs = mtc.revoke_qualification(qualification_type_id=qualification.mturk_id, subject_id=self.mturk_id, 
                                          reason=reason)
        if rs.status:
            wqs.delete()
        return rs.status
    
    def update_qualification_score(self, qualification, value=1):
        if qualification.mturk_id is None or self.mturk_id is None:
            return False
        wqs = WorkerQualification.objects.filter(worker=self, qualification=qualification)
        if wqs.count() == 0:
            return False
        wq = wqs.all()[0]
        mtc = self.account.connection(self.sandbox)
        rs = mtc.update_qualification_score(qualification_type_id=qualification.mturk_id, 
                                            worker_id=self.mturk_id, value=value)
        if rs.status:
            wq.value = value
            wq.save()
        return rs.status
    
    def block(self, reason):
        if self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        rs = mtc.block_worker(worker_id=self.mturk_id, reason=reason)
        if rs.status:
            self.is_blocked = True
            self.save()
        return rs.status
    
    def unblock(self, reason):
        if self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        rs = mtc.unblock_worker(worker_id=self.mturk_id, reason=reason)
        if rs.status:
            self.is_blocked = False
            self.save()
        return rs.status
    
    def grant_bonus(self, assignment=None, price=None, reason=None):
        if self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        if assignment is None:
            assignment = Assignment.objects.filter(worker=self).all()[0]
        rs = mtc.grant_bonus(worker_id=self.mturk_id, assignment_id=assignment.mturk_id, 
                             bonus_price=Price(price), reason=reason)
        if rs.status:
            self.is_blocked = False
            self.save()
        return rs.status
        

# The assignment of an MTurker worker to a particular HIT instance.  
class Assignment(models.Model):
    # The HIT done in this assignment
    hit = models.ForeignKey(Hit, null=True, default=None)
    
    # The worker who did this assignment
    worker = models.ForeignKey(Worker, null=True, default=None)
    
    # The response posted to MTurk
    answer = models.TextField(null=True)
    
    # One of 'Approved', 'Rejected', 'Created'
    status = models.CharField(max_length=20, db_index=True, null=True)

    # If true, the result has been stored to the database
    saved = models.BooleanField(default=False)
    
    # Assignment id when interfacing with the MTurk API
    mturk_id = models.CharField(max_length=100, db_index=True, null=True)
    
    # Time when the worker accepted this HIT
    accept_time = models.DateTimeField(auto_now_add=True)
    
    # Time when the worker submitted the response to this HIT
    submit_time = models.DateTimeField()
    
    # Optionally, one can assign a score to this assignment using a crowdsourcing algorithm
    score = models.FloatField(null=True, db_index=True)
    
    def approve(self, feedback=None):
        if self.status == 'Approved' or self.mturk_id is None:
            return False
        else:
            mtc = self.hit.hit_type.account.connection(self.hit.hit_type.sandbox)
            if self.status == 'Rejected':
                rs = mtc.approve_rejected_assignment(assignment_id=self.mturk_id, feedback=feedback)
            else:
                rs = mtc.approve_assignment(assignment_id=self.mturk_id, feedback=feedback)
            if rs.status:
                self.status = 'Approved'
                self.save()
            return rs.status
    
    def reject(self, feedback=None):
        if self.status != 'Created' or self.mturk_id is None:
            return False
        else:
            mtc = self.hit.hit_type.account.connection(self.hit.hit_type.sandbox)
            rs = mtc.reject_assignment(assignment_id=self.mturk_id, feedback=feedback)
            if rs.status:
                self.status = 'Rejected'
                self.save()
            return rs.status
    
    def grant_bonus(self, price=None, reason=None):
        return self.worker.grant_bonus(assignment=self, price=price, reason=reason)


# A customized qualification that an MTurk worker must meet to be able to do some HitType(s).
# This could be an exam that a worker must pass.  When constructing a HitType, one can
# specify an array of qualifications that must be met by workers. Note that stock qualifications 
# like Locale and Approval Rate are handled without using this class.
class Qualification(models.Model):
    # Qualification id when interfacing with the MTurk API
    mturk_id = models.CharField(max_length=100, db_index=True, null=True, default = None)
    
    # Name of this qualification type (shows up on MTurk)
    name = models.CharField(max_length=100, db_index=True, null=True)
    
    # A text description of this qualification type (shows up on MTurk)
    description = models.TextField(null=True, default = None)
    
    # Keywords tagging this qualification type (shows up on MTurk)
    keywords = models.TextField(null=True, default = None)
    
    test = models.TextField(null=True, default = None)
    
    test_duration = models.PositiveIntegerField(null=True, default = None)
    
    retry_delay = models.PositiveIntegerField(null=True, default = None)
    
    answer_key = models.TextField(null=True, default = None)
    
    auto_granted = models.BooleanField(default=False)
    
    auto_granted_value = models.PositiveIntegerField(default = 1)
    
    status = models.CharField(max_length=20, db_index=True, null=True, default='Active')
    
    # If true, this HIT will be in the sandbox instead of officially on MTurk
    sandbox = models.BooleanField(default=True)
    
    # The MTurk account used to pay for these HITs
    account = models.ForeignKey(Account, db_index=True, null=True, default=None)
    
    # The user that owns this HIT type
    user = models.ForeignKey(User, db_index=True, null=True, default=None)
    
    # TODO: validation, handle special cases, update_type, synch, url, get_qualification_requests, grant_qualification
    def register(self):
        mtc = self.account.connection(self.sandbox)
        rs = mtc.create_qualification_type(name=self.name, description=self.description, status=self.status, keywords=self.keywords, retry_delay=self.retry_delay, test=self.test, answer_key=self.answer_key, test_duration=self.test_duration, auto_granted=self.auto_granted, auto_granted_value=self.auto_granted_value)
        if rs.status:
            self.mturk_id = rs[0].QualificationTypeId  # untested
            self.save()
            return True
        return False
    
    def on_changed(self):
        if self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        rs = mtc.update_qualification_type(description=self.description, status=self.status, retry_delay=self.retry_delay, test=self.test, answer_key=self.answer_key, test_duration=self.test_duration, auto_granted=self.auto_granted, auto_granted_value=self.auto_granted_value)
        return rs.status
    
    def dispose(self):
        if self.mturk_id is None:
            return False
        mtc = self.account.connection(self.sandbox)
        rs = mtc.dispose_qualification_type(qualification_type_id=self.mturk_id)
        return rs.status
        
    

# Comments from MTurk workers pertaining to some HIT
class WorkerFeedback(models.Model):
    assignment = models.ForeignKey(Assignment, null=True, default=None)
    feedback = models.TextField(null=True)
    # TODO: create this
    

# Assignment of a particular Qualification to a particular Worker
class WorkerQualification(models.Model):
    worker = models.ForeignKey(Worker, db_index=True, null=True, default=None)
    qualification = models.ForeignKey(Qualification, db_index=True, null=True, default=None)
    value = models.PositiveIntegerField(default = 0)



